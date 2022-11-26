# Self-supervised Contrastive Representation Learning for Semi-supervised Time-Series Classification (CA-TCC) [Paper](http://arxiv.org/abs/2208.06616)

This work is built on top of [TS-TCC](https://github.com/emadeldeen24/TS-TCC), so if you are not familiar 
with it, please check it first.


Training modes:
1-CA-TCC has two new training modes over TS-TCC
- "gen_pseudo_labels": which generates pseudo labels from fine-tuned TS-TCC model. This mode assumes that you ran "ft_1per" mode first.
- "SupCon": which performs supervised contrasting on pseudo-labeled data.

Note that "SupCon" is case-sensitive.

To fine-tune or linearly evaluate "SupCon" pretrained model, include it in the training mode.
For example: "ft_1per" will fine-tune the TS-TCC pretrained model with 1% of labeled data.
"ft_SupCon_1per" will fine-tune the CA-TCC pretrained model with 1% of labeled data.
Same applies to "tl" or "train_linear".

To run everything smoothly, we included `ca_tcc_pipeline.sh` file. You can simply use it.