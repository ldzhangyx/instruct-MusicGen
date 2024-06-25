import torch
from torch import nn
from .musicgen_cc import MusicGen
from .utilities.model_utils import freeze, print_trainable_parameters
import peft


def get_musicgen(sec, device):
    mg = MusicGen.get_pretrained(name='large', device=device)
    mg.set_generation_params(duration=sec, extend_stride=16, top_k=250)
    mg.lm.here()
    freeze(mg.lm)
    return mg


class CondMusicgen(nn.Module):
    def __init__(self, sec, device="cuda", top_k=250):
        super().__init__()
        mg = get_musicgen(sec, device)
        mg.generation_params["top_k"] = top_k
        self.musicgen = mg
        self.lm = mg.lm
        self.max_duration = sec
        self.frame_rate = 50


    def set_training(self):
        self.lm.train()

    def forward(self, input_code, text_description, embed_fn, num_samples=1, mode="train",
                total_gen_len=None, prompt_tokens=None):

        mg = self.musicgen
        lm = self.lm

        # attributes, _ = mg._prepare_tokens_and_attributes(text_description, None)

        if mode == "train":
            with mg.autocast:
                out = lm.compute_predictions(codes=input_code,
                                             embed_fn=embed_fn,
                                             conditions=text_description)
            return out
        elif mode == "inference":
            if total_gen_len is None:
                total_gen_len = int(mg.duration * mg.frame_rate)

            with mg.autocast:
                gen_tokens = lm.generate(embed_fn=embed_fn, num_samples=num_samples,
                                         prompt=None, conditions=text_description,
                                         callback=None, max_gen_len=total_gen_len,
                                         **mg.generation_params)
                return gen_tokens
        elif mode == "continuation":
            with mg.autocast:
                #if prompt_tokens is not None:
                #    print(prompt_tokens.shape)
                gen_tokens = lm.generate(embed_fn=embed_fn, num_samples=num_samples,
                                         prompt=prompt_tokens, conditions=text_description,
                                         callback=None, max_gen_len=total_gen_len, **mg.generation_params)
                return gen_tokens

    def generate(self, cp_fn,
                 text_description,
                 condition_audio_code,
                 num_samples):

        mg = self.musicgen
        lm = self.lm

        attributes, _ = mg._prepare_tokens_and_attributes(text_description, None)

        all_tokens = []
        stride_tokens = int(self.frame_rate * mg.extend_stride)
        current_gen_offset = 0
        prompt_length = 0
        prompt_tokens = None
        total_gen_len = condition_audio_code.shape[-1] - 1
        total_sec = total_gen_len / 50.
        while current_gen_offset + prompt_length < total_gen_len:
            time_offset = current_gen_offset / self.frame_rate
            chunk_duration = min(total_sec - time_offset, self.max_duration)
            max_gen_len = int(chunk_duration * self.frame_rate)
            if prompt_length >= max_gen_len:
                break
            #print("current_gen_offset / total ", current_gen_offset, "/", total_gen_len)
            with mg.autocast:
                condition_audio_code_clip = condition_audio_code[:, :, current_gen_offset:current_gen_offset + max_gen_len + 1]
                #print(cond_mask.shape, drums_clip.shape, piano_roll_clip.shape, chords_clip.shape, max_gen_len)
                embed_fn = cp_fn(condition_audio_code=condition_audio_code_clip,
                                 max_n_frames=max_gen_len,
                                 mode="inference")
                gen_tokens = lm.generate(num_samples=num_samples,
                                         embed_fn=embed_fn,
                                         prompt=prompt_tokens,
                                         conditions=attributes,
                                         callback=None, max_gen_len=max_gen_len, **mg.generation_params)
            if prompt_tokens is None:
                all_tokens.append(gen_tokens)
            else:
                all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
            prompt_tokens = gen_tokens[:, :, stride_tokens:]
            prompt_length = prompt_tokens.shape[-1]
            current_gen_offset += stride_tokens
            if current_gen_offset > 50 * 80:
                break

        gen_tokens = torch.cat(all_tokens, dim=-1)
        return gen_tokens

    def get_input_embeddings(self):
        return self.lm.emb


class EmbFn:
    def __init__(self, activates, fn, start_layer, max_len, inference=False, skip=None):
        self.interval = None
        self.index = -1
        self.adaptor = None
        self.start_layer = start_layer
        self.activates = activates
        self.max_len = max_len
        self.fn = fn
        self.inference = inference
        self.skip = skip

    def get_adaptor(self, tag):
        index = self.index
        if index < self.start_layer or tag == "cross":
            return None, None
        i = index - self.start_layer
        adaptor, gate = self.fn(i, self.activates)
        # if self.adaptor is not None:
        #    adaptor = self.adaptor + adaptor
        return adaptor, gate

    def set_state(self, prefix_q, prefix_k, prefix_v):
        self.cache[str(self.index)] = [prefix_q, prefix_k, prefix_v]

    def get_state(self):
        prefix_q, prefix_k, prefix_v = self.cache[str(self.index)]
        return prefix_q, prefix_k, prefix_v

    def clear_state(self):
        self.qkv = {}
        torch.cuda.empty_cache()

    def crop(self, tag, x):

        if self.interval is not None:
            st, ed = self.interval
            if st >= self.max_len:
                st = self.max_len - 1
                ed = st + 1
            return x[:, :, st:ed, :]
        return x

    def get_cross_attention_src(self, src):
        return src

    def modify(self, x, dt_x, gate):
        # return dt_x * gate[:dt_x.shape[-2], :] + x
        return dt_x * gate + x

    def set_uncond(self, uncond):
        self.uncond = uncond

    def get_uncond(self):
        return self.uncond

    def set_uncond_cross_attention(self, x):
        self.uncond_cross_att = x

    def get_cross(self):
        return True

    def get_status(self, tag):
        return tag == "self" and self.index >= self.start_layer

    def update_adaptor(self, adaptor):
        self.adaptor = adaptor

    def set_index(self, index):
        self.index = index

    def update_interval(self, st, ed):
        self.interval = [st, ed]


class CPTransformerLayer(nn.Module):
    def __init__(self, norm1, norm2, layer_scale_1, dropout1, self_attn, layer_scale_2,
                 autocast, linear1, linear2, activation, dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm1 = norm1
        self.norm2 = norm2
        self.layer_scale_1 = layer_scale_1
        self.dropout1 = dropout1
        self.self_attn = self_attn
        self.layer_scale_2 = layer_scale_2
        self.autocast = autocast
        self.linear1 = linear1
        self.linear2 = linear2
        self.activation = activation
        self.dropout = dropout

    def _ff_block(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

    def forward(self, x, cond=None):
        with self.autocast:
            if cond is None:
                nx = self.norm1(x)
            else:
                nx = self.norm1(x) + cond
            q, k, v, o = self.self_attn(nx, nx, nx, emb_fn=None,
                                        attn_mask=None,
                                        key_padding_mask=None,
                                        need_weights=False, is_causal=False, return_qkv=True)
            x = x + self.layer_scale_1(self.dropout1(o))
            x = x + self.layer_scale_2(self._ff_block(self.norm2(x)))
        return q, k, v, x


class CPTransformer(nn.Module):
    def __init__(self, model, emb_fn, start_layer, latent_dim, autocast, stride=50 * 10):
        super().__init__()

        self.emb_fn = {
            "emb": emb_fn
        }

        new_layers = nn.ModuleList()

        hidden_dim = 2048
        cond_dim = latent_dim
        num_layers = len(model.layers) - start_layer
        max_n_frames = 500

        self.pos_emb = nn.Parameter(
            torch.randn(num_layers + 1, max_n_frames + 1, hidden_dim),
            requires_grad=True)
        # self.encodec_emb = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.merge_linear = nn.ModuleList()
        # self.piano_roll_emb = nn.ModuleList()
        for i in range(start_layer, len(model.layers)):
            norm1 = model.layers[i].norm1
            norm2 = model.layers[i].norm2
            layer_scale_1 = model.layers[i].layer_scale_1
            dropout1 = model.layers[i].dropout1
            self_attn = model.layers[i].self_attn
            layer_scale_2 = model.layers[i].layer_scale_2
            linear1 = model.layers[i].linear1
            linear2 = model.layers[i].linear2
            activation = model.layers[i].activation
            dropout = model.layers[i].dropout
            new_layers.append(CPTransformerLayer(norm1=norm1,
                                                 norm2=norm2,
                                                 layer_scale_1=layer_scale_1,
                                                 dropout1=dropout1,
                                                 self_attn=self_attn,
                                                 linear1=linear1,
                                                 linear2=linear2,
                                                 activation=activation,
                                                 dropout=dropout,
                                                 layer_scale_2=layer_scale_2,
                                                 autocast=autocast))

            self.merge_linear.append(nn.Linear(cond_dim, hidden_dim, bias=False))
            # self.piano_roll_emb.append(nn.Linear(128, latent_dim, bias=False))

        self.layers = new_layers
        # self.gates = nn.Parameter(torch.zeros([num_layers, max_n_frames, 64]))
        self.gates = nn.Parameter(torch.zeros([num_layers]))
        freeze(self.layers)

        self.max_n_frames = max_n_frames
        self.start_layer = start_layer
        self.num_layers = num_layers
        self.stride = stride

    def fn(self, i, activates):
        if i >= self.num_layers:
            return None, None
        return activates[i]

    def forward(self, condition_audio_code, max_n_frames, mode, skip=None):
        max_n_frames = self.max_n_frames if max_n_frames is None else max_n_frames

        sum_code = sum([self.emb_fn["emb"][i](condition_audio_code[:, i]) for i in range(4)])


        condition_audio_code = sum_code
        # condition_audio_code = self.encodec_emb(sum_code)

        B, T, latent_dim = condition_audio_code.shape  # (batch_size, n_frames, latent_dim)

        o = self.pos_emb[0][None, :T].repeat(B, 1, 1)
        #print(o.shape, T)

        outs = []

        encoded_condition = condition_audio_code
        for i in range(len(self.layers)):
            # We conduct two pass transformer.
            # The first pass is to get multi-layer representation of the condition_audio_code.
            # The second pass is to fuse the condition back.

            # 1st pass
            # encoded_condition = encoded_condition + self.pos_emb[i + 1][None, :T].repeat(B, 1, 1)
            # _, _, _, encoded_condition = self.layers[i](x=encoded_condition, cond=None)

            # add positional encoding and send to the transformer
            embedding = self.merge_linear[i](encoded_condition) + self.pos_emb[i + 1][None, :T].repeat(B, 1, 1)
            # embedding = (encoded_condition) + self.pos_emb[i + 1][None, :T].repeat(B, 1, 1) + self.merge_linear[i](condition_audio_code)

            # 2nd pass
            q, k, v, o = self.layers[i](x=o, cond=embedding)  #
            if not mode == "train":
                outs.append([[torch.cat([q, q], 0),
                              torch.cat([k, k], 0),
                              torch.cat([v, v], 0)], self.gates[i]])
            else:
                outs.append([[q, k, v], self.gates[i]])

        emb_fn = EmbFn(activates=outs, fn=self.fn,
                       start_layer=self.start_layer,
                       max_len=max_n_frames,
                       inference=(mode == "inference"),
                       skip=skip)

        return emb_fn

    def save_weights(self, path):
        state_dict = {}
        sdict = self.state_dict()
        for n in sdict:
            if str.startswith(n, "layers"):
                continue
            state_dict[n] = sdict[n]
        torch.save(state_dict, path)

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)


class Instructor(nn.Module):
    def __init__(self, sec, num_layers, latent_dim, top_k):
        '''The MusicGen model with instructor adapter.

        Args:
            sec: int, duration of the audio in seconds
            num_layers: int, number of layers of adapter in the transformer
            latent_dim: int, dimension of the latent space
        '''
        super().__init__()
        lm = CondMusicgen(sec, top_k=top_k)
        self.peft_model = lm
        self.musicgen = lm.musicgen
        self.cp_transformer = CPTransformer(self.musicgen.lm.transformer,
                                            emb_fn=self.musicgen.lm.emb,
                                            start_layer=48 - num_layers,
                                            latent_dim=latent_dim,
                                            autocast=self.musicgen.autocast)
        self.text_lora_config = peft.LoraConfig(target_modules=r".*\.cross_attention\.(q_proj|v_proj)",
                                                r=32,
                                                lora_alpha=64)

        self.peft_model.lm.transformer = peft.get_peft_model(self.peft_model.lm.transformer, self.text_lora_config)

    def set_training(self):
        self.peft_model.set_training()
        print_trainable_parameters(self)

    def save_weights(self, path):
        self.cp_transformer.save_weights(path)

    def load_weights(self, path):
        self.cp_transformer.load_weights(path)

    def forward(self, input_code, text_description, condition_audio_code,
                num_samples=8, mode="train", max_n_frames=None, prompt_tokens=None):

        if max_n_frames is None:
            max_n_frames = input_code.shape[-1]

        condition_audio_code = torch.cat([condition_audio_code, torch.ones_like(condition_audio_code[:,:, 0:1]) * 2048], dim=-1)

        embed_fn = self.cp_transformer.forward(condition_audio_code=condition_audio_code,
                                       max_n_frames=max_n_frames,
                                       mode=mode,
                                       skip=None)
        out = self.peft_model.forward(input_code,
                              text_description=text_description,
                              embed_fn=embed_fn,
                              mode=mode,
                              total_gen_len=max_n_frames,
                              prompt_tokens=prompt_tokens)
        return out

    def generate(self, text_description, condition_audio_code,
                 num_samples=1):
        out = self.peft_model.generate(cp_fn=self.cp_transformer,
                                       text_description=text_description,
                                       condition_audio_code=condition_audio_code,
                                       num_samples=num_samples,
                                       )
        return out
