import torch
import numpy as np
class TransformerDataset:
    def __init__(self, texts, author_ids, src_tokenizer, trg_tokenizer, MAX_LEN):
        
        self.texts = texts
        self.author_ids = author_ids
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.MAX_LEN = MAX_LEN
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        author_id = self.author_ids[item]
        
        src = torch.LongTensor(self.src_tokenizer.encode(text, padding='max_length', max_length=self.MAX_LEN)[:self.MAX_LEN])
        speaker_id = torch.LongTensor([author_id])
        decoder_src = torch.LongTensor(self.trg_tokenizer.encode(text, padding='max_length', max_length=self.MAX_LEN)[:self.MAX_LEN])
        
        return {"input_ids": src, "speaker_ids": speaker_id, "decoder_input_ids": decoder_src}
                
class PreTrainedTransformerDataset:
    def __init__(self, texts, tokenizer, MAX_LEN):
        
        self.texts = texts
        self.tokenizer = tokenizer
        self.MAX_LEN = MAX_LEN
        self.eos_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]

        zero_src = torch.LongTensor(np.zeros(self.MAX_LEN))
        if self.tokenizer.__class__.__name__ not in ['XLNetTokenizerFast','BertTokenizerFast']:
            if self.eos_token:
                text += self.eos_token
            else:
                text += self.tokenizer.sep_token

        src = torch.LongTensor(self.tokenizer.encode(text)[:self.MAX_LEN])
        zero_src[:src.size()[0]] = src

        return {"input_ids": zero_src}
         