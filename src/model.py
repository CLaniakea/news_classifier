import torch
import torch.nn as nn
import pdb

class news_classifier(nn.Module):
    def __init__(self,vocab_size, in_dim, hid_dim, num_cls, num_layers, num_heads) -> None:
        super(news_classifier, self).__init__()
        self.embeds = nn.Embedding(vocab_size, in_dim, max_norm=True)
        Transformer_layer=nn.TransformerEncoderLayer(d_model=in_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(Transformer_layer, num_layers=num_layers)
        self.cls=nn.Linear(hid_dim, num_cls)
    
    def forward(self, data, mask):
        # data : [b,s,d]
        # mask : [b,s]
        #pdb.set_trace()
        embeds = self.embeds(data)
        #pdb.set_trace()
        embeds=embeds.transpose(0,1)
        #mask=mask.transpose(0,1)
        e_out=self.encoder(embeds, mask=None,src_key_padding_mask=mask)
        out=self.cls(e_out)
        # out=torch.mean(out,0)
        # out=out.squeeze(0)
        out=torch.max(out,0)[0]
        #out=out.transpose(0,1)
        return torch.softmax(out, 1)
