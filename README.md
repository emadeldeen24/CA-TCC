# Self-supervised Contrastive Representation Learning for Semi-supervised Time-Series Classification (CA-TCC) [[Paper](http://arxiv.org/abs/2208.06616)] [[Cite](#citation)]

## This work is an extention to [TS-TCC](https://github.com/emadeldeen24/TS-TCC), so if you need any details about the unsupervised pretraining and/or the datasets and its preprocessing, please check it first.


### Training modes:
CA-TCC has two new training modes over TS-TCC
- "gen_pseudo_labels": which generates pseudo labels from fine-tuned TS-TCC model. This mode assumes that you ran "ft_1per" mode first.
- "SupCon": which performs supervised contrasting on pseudo-labeled data.

Note that "SupCon" is case-sensitive.

To fine-tune or linearly evaluate "SupCon" pretrained model, include it in the training mode.
For example: "ft_1per" will fine-tune the TS-TCC pretrained model with 1% of labeled data.
"ft_SupCon_1per" will fine-tune the CA-TCC pretrained model with 1% of labeled data.
Same applies to "tl" or "train_linear".

### Training procedure
To run everything smoothly, we included `ca_tcc_pipeline.sh` file. You can simply use it.


## Citation
If you found this work useful for you, please consider citing it.
```
@inproceedings{ijcai2021-324,
  title     = {Time-Series Representation Learning via Temporal and Contextual Contrasting},
  author    = {Eldele, Emadeldeen and Ragab, Mohamed and Chen, Zhenghua and Wu, Min and Kwoh, Chee Keong and Li, Xiaoli and Guan, Cuntai},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, {IJCAI-21}},
  pages     = {2352--2359},
  year      = {2021},
}
```
```
@article{emadeldeen2022catcc,
  title   = {Self-supervised Contrastive Representation Learning for Semi-supervised Time-Series Classification},
  author  = {Eldele, Emadeldeen and Ragab, Mohamed and Chen, Zhenghua and Wu, Min and Kwoh, Chee Keong and Li, Xiaoli and Guan, Cuntai},
  journal = {arXiv preprint arXiv:2208.06616},
  year    = {2022}
}
```

## Contact
For any issues/questions regarding the paper or reproducing the results, please contact me.   
Emadeldeen Eldele   
School of Computer Science and Engineering (SCSE),   
Nanyang Technological University (NTU), Singapore.   
Email: emad0002{at}e.ntu.edu.sg   
