from __future__ import absolute_import, division, print_function

import os

import torch
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

from model import VPLM, VConfig




def main():
    config = AutoConfig.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased",
                                                          from_tf=bool('.ckpt' in "bert-base-uncased"),
                                                          config=config)
    hplm = model.load_state_dict(torch.load("result/H-PLM_bert/HPLM.bin", map_location='cpu'), strict=False)

    model_to_save = hplm.module if hasattr(hplm, 'module') else hplm
    output_dir = os.path.join('result/H-PLM_bert', 'checkpoint-best')
    model_to_save.save_pretrained('result/H-PLM_bert' )
    tokenizer.save_pretrained('result/H-PLM_bert')
    
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased",
                                                          from_tf=bool('.ckpt' in "bert-base-uncased"),
                                                          config=config)

    tplm = model.load_state_dict(torch.load("result/T-PLM_bert/HPLM.bin", map_location='cpu'), strict=False)

    model_to_save = tplm.module if hasattr(tplm, 'module') else tplm
    output_dir = os.path.join('result/T-PLM_bert', 'checkpoint-best')
    model_to_save.save_pretrained('result/T-PLM_bert' )
    tokenizer.save_pretrained('result/T-PLM_bert')

    
    
    bert_config = config
                                                     
    bert_model = model
    bert_model.resize_token_embeddings(len(tokenizer))
    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased",
                                                          from_tf=bool('.ckpt' in "bert-base-uncased"),
                                                          config=config)
    html_config = VConfig(args, **config.__dict__)
    model = VPLM(bert_model, html_config)
    model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model.bin')))  # confirmed correct

    vplm = model.load_state_dict(torch.load("result/V-PLM_bert/HPLM.bin", map_location='cpu'), strict=False)

    model_to_save = vplm.module if hasattr(vplm, 'module') else vplm
    output_dir = os.path.join('result/V-PLM_bert', 'checkpoint-best')
    model_to_save.save_pretrained('result/V-PLM_bert' )
    tokenizer.save_pretrained('result/V-PLM_bert')



if __name__ == "__main__":
    main()
    


