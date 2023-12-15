# Agent Attention for Stable Diffusion

When applied to Stable Diffusion, our agent attention accelerates generation and substantially enhances image generation quality **without any additional training**.


## AgentSD
We practically apply agent attention to [ToMeSD model](https://github.com/dbolya/tomesd). ToMeSD reduces the number of tokens before attention calculation in Stable Diffusion, enhancing generation speed. Nonetheless, the post-merge token count remains considerable, resulting in continued complexity and latency. Hence, we replace the Softmax attention employed in ToMeSD model with our agent attention to further enhance speed. 

Here are results with [FID](https://github.com/mseitzer/pytorch-fid) scores vs. time and memory usage (lower is better) when employing Stable Diffusion v1.5 to generate 2,000 $512^2$ images of ImageNet-1k classes using 50 PLMS diffusion steps on a single RTX4090 GPU:

| Method                | r%   | FID ↓                      | Time (s/im) ↓             | Memory (GB/im) ↓        |
| --------------------- | ---- | :------------------------- | :------------------------ | :---------------------- |
| Stable Diffusion v1.5 | 0    | 28.84 (_baseline_)         | 2.62 (_baseline_)         | 3.13 (_baseline_)       |
| ToMeSD                | 10   | 28.64                      | 2.40                      | 2.55                    |
|                       | 20   | 28.68                      | 2.15                      | 2.03                    |
|                       | 30   | 28.82                      | 1.90                      | 2.09                    |
|                       | 40   | 28.74                      | 1.71                      | 1.69                    |
|                       | 50   | 29.01                      | 1.53                      | 1.47 |
| **AgentSD**           | 10   | 27.79 (↓**1.05** _better_) | 1.97 (**1.33x** _faster_) | 1.77 (**1.77x** _less_) |
|                       | 20   | 27.77 (↓**1.07** _better_) | 1.80 (**1.45x** _faster_) | 1.60 (**1.95x** _less_) |
|                       | 30   | 28.03 (↓**0.81** _better_) | 1.65 (**1.59x** _faster_) | 2.05 (**1.53x** _less_) |
|                       | 40   | 28.15 (↓**0.69** _better_) | 1.54 (**1.70x** _faster_) | 1.55 (**2.02x** _less_) |
|                       | 50   | 28.42 (↓**0.42** _better_) | 1.42 (**1.84x** _faster_) | 1.21 (**2.59x** _less_) |

ToMeSD accelerates Stable Diffusion while maintaining similar image quality. AgentSD not only **further accelerates** ToMeSD but also **significantly enhances** image generation quality **without extra training!**

## Dependencies

- PyTorch >= 1.12.1


## Usage
Place the [agentsd](./) folder in your project and apply AgentSD to any Stable Diffusion model with:
```py
import agentsd
if step == 0:
	# Apply Agent Attention and ToMe during early 20 diffusion steps
    agentsd.apply_patch(model, sx=4, sy=4, ratio=0.4, agent_ratio=0.95)
elif step == 20:
	# Apply ToMe in later diffusion steps
	agentsd.remove_patch(model)
	agentsd.apply_patch(model, sx=2, sy=2, ratio=0.4, agent_ratio=0.5)
```
### Example
To apply AgentSD to SDv1 PLMS sampler, add the following to [this line](https://github.com/runwayml/stable-diffusion/blob/08ab4d326c96854026c4eb3454cd3b02109ee982/ldm/models/diffusion/plms.py#L143):
```py
import agentsd
if i == 0:
    agentsd.apply_patch(self.model, sx=4, sy=4, ratio=0.4, agent_ratio=0.95)
elif i == 20:
	agentsd.remove_patch(self.model)
	agentsd.apply_patch(self.model, sx=2, sy=2, ratio=0.4, agent_ratio=0.5)
```
To apply AgentSD to SDv2 DDIM sampler, add the following to [this line](https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/models/diffusion/ddim.py#L152) (setting ``attn_precision="fp32"`` to avoid [numerical instabilities on the v2.1 model](https://github.com/Stability-AI/stablediffusion/tree/main?tab=readme-ov-file#news)):

```py
import agentsd
if i == 0:
    agentsd.apply_patch(self.model, sx=4, sy=4, ratio=0.4, agent_ratio=0.95, attn_precision="fp32")
elif i == 20:
	agentsd.remove_patch(self.model)
	agentsd.apply_patch(self.model, sx=2, sy=2, ratio=0.4, agent_ratio=0.5, attn_precision="fp32")
```

## TODO

 - [x] [Stable Diffusion v1](https://github.com/runwayml/stable-diffusion)
 - [x] [Stable Diffusion v2](https://github.com/Stability-AI/stablediffusion)
 - [x] [Latent Diffusion](https://github.com/CompVis/latent-diffusion)
 - [ ] [Diffusers](https://github.com/huggingface/diffusers)

## Citation

If you find this repo helpful, please consider citing us.

```latex
@article{han2023agent,
  title={Agent Attention: On the Integration of Softmax and Linear Attention},
  author={Han, Dongchen and Ye, Tianzhu and Han, Yizeng and Xia, Zhuofan and Song, Shiji and Huang, Gao},
  journal={arXiv preprint arXiv:2312.08874},
  year={2023}
}
```
