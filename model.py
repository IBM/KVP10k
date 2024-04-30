import torch
import transformers
import lightning.pytorch as pl
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model
from tokenizers.processors import TemplateProcessing
from utils.config import Config


class Model(pl.LightningModule):
    def __init__(self, args: Config):
        super().__init__()
        self.args = args
        self.save_hyperparameters(vars(args))

        # tokenizer
        print('Tokenizer is ', args.tokenizer)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(**args.tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.bos_token = self.tokenizer.eos_token
        self.tokenizer._tokenizer.post_processor = TemplateProcessing(
            single="[bos] $A [eos]",
            special_tokens=(
                ("[bos]", self.tokenizer.bos_token_id),
                ("[eos]", self.tokenizer.eos_token_id)
            ),
        )

        # LLM
        config = transformers.AutoConfig.from_pretrained(**args.llm)
        if args.config is not None:
            config.update(args.config)

        print("Using bfloat16 in LLM")
        self.llm = transformers.AutoModelForCausalLM.from_pretrained(**args.llm,
                                                                     torch_dtype=torch.bfloat16,
                                                                     config=config)

        self.generation_config = transformers.GenerationConfig(
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # peft (parameter-efficient fine-tuning)
        if args.peft is not None:
            peft_config = LoraConfig(**args.peft)
            self.llm = get_peft_model(self.llm, peft_config)
            self.llm.print_trainable_parameters()

        # load pretrained if exists
        self.load_pretrained_weights()

        print(self)

    def forward(self, prompt, image=None, context=''):
        with torch.inference_mode():
            batch = {'prompt': [prompt], 'target': [''], 'context': [context]}

            llm_inputs = self.get_llm_inputs(batch, max_length=8192)
            # remove labels
            del llm_inputs['labels']
            # remove eos_token
            if 'inputs_embeds' in llm_inputs:
                llm_inputs['inputs_embeds'] = llm_inputs['inputs_embeds'][:, :-1, :]
            else:
                llm_inputs['input_ids'] = llm_inputs['input_ids'][:, :-1]
            llm_inputs['attention_mask'] = llm_inputs['attention_mask'][:, :-1]

            response_ids = self.llm.generate(**llm_inputs,
                                             max_new_tokens=self.args.inference_max_new_tokens,
                                             temperature=0.0,
                                             generation_config=self.generation_config)
            response = self.tokenizer.decode(response_ids.squeeze())

            if 'inputs_embeds' in llm_inputs:  # embeddings were used as input, simply remove bos_token
                start_idx = len(self.tokenizer.bos_token)
            else:  # token indices were used as input, remove entire text prompt
                start_idx = response.find("### Response:\n") + len("### Response:\n")
            response = response[start_idx:]
            response = response[:response.find(self.tokenizer.eos_token)]  # remove text after eos token

            # debug
            # print('*' * 10, 'response:', '\n', response)
            return response

    def _step(self, batch, max_length, padding):
        llm_inputs = self.get_llm_inputs(batch, max_length, padding)
        if (llm_inputs['labels'] == -100).all():  # response has been truncated completely, do not run model
            return None
        return self.llm(**llm_inputs)['loss']

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, self.args.train_max_length, self.args.train_padding)
        if loss is not None:
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                     batch_size=self.args.train_batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, self.args.val_max_length, self.args.val_padding)
        if loss is not None:
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True,
                     batch_size=self.args.val_batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step(batch, self.args.val_max_length, self.args.val_padding)
        if loss is not None:
            self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.args.optimizer_8bit:
            return bnb.optim.Adam8bit(self.parameters(), lr=self.args.optimizer_lr)
        return torch.optim.AdamW(self.parameters(), lr=self.args.optimizer_lr)



    def get_llm_inputs(self, batch, max_length=None, padding=True):

        # get prompts (with response)
        prompts = [self.make_prompt(instruction, input_, output)
                   for instruction, input_, output in zip(batch['prompt'], batch['context'], batch['target'])]

        # tokenize
        encoding = self.tokenizer(prompts,
                                  max_length=max_length,
                                  padding=padding,
                                  truncation=True,
                                  return_tensors='pt').to(self.device)

        input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']
        if max_length is not None and input_ids.shape[1] >= max_length - 2:
            print('-------------------')
            print('Input size is ',input_ids.shape[1], ' when max length is', max_length)

        labels = input_ids.clone()

        # replace all padding tokens with -100 so it's ignored by the cross entropy loss
        labels[attention_mask == 0] = -100

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


    def load_pretrained_weights(self):
        if self.args.pretrained is not None:
            print("=> loading checkpoint '{}'".format(self.args.pretrained))
            checkpoint = torch.load(self.args.pretrained)
            state_dict = checkpoint['state_dict']

            if self.args.load_only_mm_alignment: # do not load LLM or VisionEncoder
                for k in list(state_dict.keys()):
                    if 'llm' in k and 'norm' in k:  # FIXME: always load norm layers for now
                        pass
                    elif 'llm' in k or 'vision_encoder' in k:
                        del state_dict[k]

            if self.args.peft is not None:
                for k in list(state_dict.keys()):
                    if k.startswith('llm.') and not k.startswith('llm.base_model.model.'):
                        state_dict['llm.base_model.model.' + k[len('llm.'):]] = state_dict[k]
                        del state_dict[k]

            msg = self.load_state_dict(state_dict, strict=False)
            print('*' * 50 + '\n', msg, '\n' + '*' * 50)
            print("=> loaded pre-trained model '{}'".format(self.args.pretrained))

    @staticmethod
    def make_prompt(instruction, input_, output=""):

        return "{0}\n{1}\n{2}\n{3}\n{4}\n{5}\n{6}\n{7}\n{8}\n{9}\n{10}\n{11}\n{12}".format(
            "<Document>",  # 0
            input_,  # 1
            "</Document>",  # 2
            "<Task>",  # 3
            "From the document, extract the text keys and values. ", # 4
            "Please provide the response in the form of a Python list of lists. ", #5
            "It should begin with “[“ and end with “]”", # 6
            "Each internal list should contain two comma separated string items - key and value, and should begin with “[“ and end with “]” as well. ", # 7
            "The example of format is as following:",  # 8
            "[[\"key1\":\"value1\"], [\"key2\":\"value2\"], [\"key3\":\"value3\"]]", # 9
            "</Task>",   # 10
            "### Response:",  # 11
            output,  # 12
        )



