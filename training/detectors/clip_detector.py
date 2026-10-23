
import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from metrics.base_metrics_class import calculate_metrics_for_train
from collections import OrderedDict
import clip
from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
import loralib as lora
import copy
from typing import Union, List
from scl import SingleCenterLoss
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES
from fre_networks.freqnet import freqnet



_tokenizer = _Tokenizer()
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='clip')
class CLIPDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model,self.processor = self.build_backbone(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_func = self.build_loss(config)
        self.logit_scale = nn.Parameter((torch.ones([])) * np.log(1/0.07))
        # self.a = nn.Parameter(torch.tensor(0.2),requires_grad=True)
        self.head = nn.Linear(512, 768).to(self.device)
        self.head_nose = nn.Linear(512, 768).to(self.device)
        self.head_eye = nn.Linear(512, 768).to(self.device)

        self.classnames_real = ['real_human_face']
        self.classnames_fake = ['fake_human_face']
        torch.nn.init.normal_(self.head.weight, std=0.01)
        torch.nn.init.normal_(self.head_nose.weight, std=0.01)

        self.ctx_init_real  = 'This is a natural facial image. The details in the image fully match the traits of a real human face, with no traces of "feature blending from multiple identities". The shadows on the face transition gently, showing consistent lighting effects. The contour around the eyes, the shape of the mouth, and the overall facial outline are all natural and smooth, with no unnatural distorted parts;  the color at the junction of the chin contour and the mouth is also consistent, with no incongruous jumps, fully presenting the real texture and color distribution of facial skin.'
        self.ctx_init_fake = 'This is a morphed facial image. The noise and artifacts in the image all indicate the presence of "feature blending from multiple identities". There are abrupt transitions in the shadows on the face, with inconsistent lighting that does not conform to real-world optical principles. The contour around the eyes ,the mouse and the overall facial outline have unnatural, distorted areas, and there are noticeable artifacts that disrupt the natural appearance of the face. The color transition at the junction of the forehead and the eyes, as well as at the junction of the chin contour and the mouth, is very incongruous'
        # self.ctx_init_real ='a photo of a'
        # self.ctx_init_fake = 'a photo of a'
        self.prompt_learner_real = VLPromptLearner(self.config, self.classnames_real, self.model,self.ctx_init_real)
        self.prompt_learner_fake = VLPromptLearner(self.config, self.classnames_fake, self.model,self.ctx_init_fake)
        self.tokenized_prompts_real = self.prompt_learner_real.tokenized_prompts
        self.tokenized_prompts_fake = self.prompt_learner_fake.tokenized_prompts
        self.text_encoder = TextEncoder(self.model)
        self.image_encoder = self.model.visual
        self.fre_model = freqnet()
        # self._init_frequency_model()
        self.ATAM_img = WeightedAggregator(output_seq_len=40,input_feat_dim=768).cuda()
        self.ATAM_text = WeightedAggregator(output_seq_len=50,input_feat_dim=512,proj = False).cuda()

    def build_backbone(self, config):
        # prepare the backbone
        model,processor = get_clip(
            model_name="/media/gpu/1cd8a4a1-b4ef-4f56-ac4d-10dff3dc7b13/zzx4985/DeepfakeBench/DeepfakeBench-main/pretrained_model/ViT-B-16.pt",zero_shot_model=False)
        return model,processor

        
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func


    def features(self, data_dict: dict,shared_ctx, deep_compound_prompts_vision) -> torch.tensor:
        mouse_features_tensor = self.image_encoder(data_dict['mouse'], shared_ctx, deep_compound_prompts_vision)
        class_token = self.head(mouse_features_tensor)
        img_features_tensor = self.image_encoder(data_dict['image'], shared_ctx, deep_compound_prompts_vision,
                                                 class_token)
        mouse_features_tensor = mouse_features_tensor / self.get_vector_norm(mouse_features_tensor)
        img_features_tensor = img_features_tensor / self.get_vector_norm(img_features_tensor)
        return  img_features_tensor

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred_img_eye = pred_dict['cls_img_eye']
        pred_img_nose = pred_dict['cls_img_nose']
        pred_img_mouse = pred_dict['cls_img_mouse']
        pred_img_person = pred_dict['cls']

        loss_part_eye = self.loss_func(pred_img_eye, label)
        loss_part_nose = self.loss_func(pred_img_nose, label)
        loss_part_mouse = self.loss_func(pred_img_mouse, label)
        loss_part_person = self.loss_func(pred_img_person, label)

        loss_scl_text = F.l1_loss(pred_dict['text_outputs'], pred_dict['fixed_embeddings'],
                                  reduction='mean') * 20
        # Calculate the L_SCL_image loss


        L_SCL = ( loss_scl_text)


        loss = loss_part_person+loss_part_mouse+L_SCL+loss_part_nose+loss_part_eye

        loss_dict = {'overall': loss}
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict

    def get_vector_norm(self,tensor: torch.Tensor) -> torch.Tensor:
        """
        This method is equivalent to tensor.norm(p=2, dim=-1, keepdim=True) and used to make
        model `executorch` exportable. See issue https://github.com/pytorch/executorch/issues/3566
        """
        square_tensor = torch.pow(tensor, 2)
        sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
        normed_tensor = torch.pow(sum_tensor, 0.5)
        return normed_tensor


    def forward(self, data_dict: dict, inference=False) -> dict:
        tokenized_prompts_real = self.tokenized_prompts_real
        tokenized_prompts_fake = self.tokenized_prompts_fake
        prompts_real = self.prompt_learner_real()
        prompts_fake = self.prompt_learner_fake()

        text_features_real= self.text_encoder(prompts_real, tokenized_prompts_real)
        text_features_fake = self.text_encoder(prompts_fake, tokenized_prompts_fake)

        text_features = torch.cat((text_features_real, text_features_fake), dim=0)
        text_outputs = text_features/self.get_vector_norm(text_features)


        nose_features_tensor,_ = self.image_encoder(data_dict['nose'].to('cuda'))
        nose_token = self.head_nose(nose_features_tensor)
        eye_features_tensor,_ = self.image_encoder(data_dict['eye'].to('cuda'),nose_token)
        eye_token = self.head_nose(eye_features_tensor)
        mouse_features_tensor,_ = self.image_encoder((data_dict['mouse'].to('cuda')),eye_token)
        class_token = self.head(mouse_features_tensor)
        img_features_tensor,_ = self.image_encoder((data_dict['image'].to('cuda')),class_token)
        mouse_features_tensor = mouse_features_tensor / self.get_vector_norm(mouse_features_tensor)
        img_features_tensor = img_features_tensor / self.get_vector_norm(img_features_tensor)
        nose_features_tensor = nose_features_tensor / self.get_vector_norm(nose_features_tensor)
        eye_features_tensor = eye_features_tensor / self.get_vector_norm(eye_features_tensor)
        logit_scale = self.logit_scale.exp()
        logits_per_image_mouse = (mouse_features_tensor @ text_outputs.T)*logit_scale
        logits_per_image_person = (img_features_tensor @ text_outputs.T)*logit_scale
        logits_per_image_nose = (nose_features_tensor @ text_outputs.T)*logit_scale
        logits_per_image_eye = (eye_features_tensor @ text_outputs.T)*logit_scale
        fixed_embeddings_real = self.prompt_learner_real.fixed_embeddings
        fixde_embeddings_fake = self.prompt_learner_fake.fixed_embeddings
        fixed_embeddings = torch.cat((fixed_embeddings_real, fixde_embeddings_fake), dim=0)
        fixed_embeddings = fixed_embeddings / self.get_vector_norm(fixed_embeddings)
        pred = logits_per_image_person+0.5*logits_per_image_mouse + 0.5*logits_per_image_nose + 0.5*logits_per_image_eye
        prob_person = torch.softmax(pred, dim=1)[:, 1]
        return { 'cls':logits_per_image_person,'cls_img_mouse': logits_per_image_mouse,'cls_img_nose':logits_per_image_nose,'cls_img_eye':logits_per_image_eye,
                 'mouse_features_tensor':mouse_features_tensor,'img_features_tensor':img_features_tensor,
                 'text_outputs':text_outputs,

                 'fixed_embeddings': fixed_embeddings,
                 'prob':prob_person,
                'feat': img_features_tensor,

                }
class WeightedAggregator(nn.Module):
    def __init__(self, output_seq_len,input_feat_dim,proj = True,
                  output_feat_dim=512,
                 key_dim=128, tau=0.1, activation=nn.GELU()):
        super().__init__()
        self.tau = tau
        self.activation = activation


        self.W_k = nn.Parameter(torch.randn(input_feat_dim, key_dim))  # [768, K=128]
        self.W_q = nn.Parameter(torch.randn(output_seq_len, key_dim))  # [40, K=128]


        self.W_proj = nn.Parameter(torch.randn(input_feat_dim, output_feat_dim))  # [768, 512]
        self.proj = proj

        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_proj)

    def forward(self, X):
        # X: [32, 200, 768]


        X_Wk = X @ self.W_k  # [32, 200, 768] @ [768, 128] → [32, 200, 128]

        sigma_XWk = self.activation(X_Wk)  # [32, 200, 128]

        sigma_XWk_T = sigma_XWk.permute(0, 2, 1)  # [32, 128, 200]

        similarity = self.W_q @ sigma_XWk_T  # [40, 128] @ [32, 128, 200] → [32, 40, 200]

        W_ref = F.softmax(similarity / self.tau, dim=-1)

        Y = W_ref @ X  # [32, 40, 200] @ [32, 200, 768] → [32, 40, 768]
        if self.proj == True:
            Y = Y @ self.W_proj  # [32, 40, 768] @ [768, 512] → [32, 40, 512]

        return Y

class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model,ctx_init):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg['TRAINER']['PROMPTSRC']['PROMPT_DEPTH_TEXT'] >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg['TRAINER']['PROMPTSRC']['N_CTX_TEXT']
        # ctx_init = cfg['TRAINER']['PROMPTSRC']['CTX_INIT']
        dtype = clip_model.dtype
        vis_dim = clip_model.visual.output_dim
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize =  224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and n_ctx <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                prompt = prompt.to('cuda')
                embedding = clip_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp, _ = get_clip(model_name="/media/gpu/1cd8a4a1-b4ef-4f56-ac4d-10dff3dc7b13/zzx4985/DeepfakeBench/DeepfakeBench-main/pretrained_model/ViT-B-16.pt", zero_shot_model=True)

        with torch.no_grad():
            tokenized_prompts = tokenized_prompts.to('cuda')
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                # print(x_tokenized.shape)
                text_features = clip_model_temp.encode_text(x_tokenized.cuda())
                all_teacher_features.append(text_features.unsqueeze(1))

        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.positional_embedding_res = clip_model.positional_embedding_res
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.mask1 = torch.zeros([248, 1])
        self.mask1[:20, :] = 1
        self.mask2 = torch.zeros([248, 1])
        self.mask2[20:, :] = 1

    def forward(self, prompts, tokenized_prompts):
        x = prompts + (self.positional_embedding.to(prompts.device) * self.mask1.to(prompts.device)).type(self.dtype).to(prompts.device) + (self.positional_embedding_res.to(prompts.device) * self.mask2.to(prompts.device)).type(self.dtype).to(prompts.device)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


def get_clip(model_name = "/media/gpu/1cd8a4a1-b4ef-4f56-ac4d-10dff3dc7b13/zzx4985/DeepfakeBench/DeepfakeBench-main/pretrained_model/ViT-B-16.pt",
             zero_shot_model=False):
    if not zero_shot_model:
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 9,
                          "language_depth": 9,
                          "vision_ctx": 4,
                          "language_ctx": 4}
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0,
                          "vision_ctx": 0,
                          "language_ctx": 0}
    model,processor =clip.load(
        name=model_name,
        design_details=design_details,
        device="cuda",
    )
    for param in model.parameters():
        param.requires_grad = True
    return  model,processor