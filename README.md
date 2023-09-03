# explainable-RGCNs-and-RGATs

This repository documents the implementation on using LRP with RGCNs and RGATs described in my Master's Thesis. 
In includes the preprocessing of the datasets and embeddings, the implemention of both models with LRP as well as the evaluation of all models.


## Getting Started

### Dependencies

* To run the models on GPU, the environment listed in `gpu_rgcnrgatlrp.yml`are used.
* To run the models on CPU, the environment listed in `cpu_rgcnrgatlrp.yml`are used.

### Installing

* To get the IMDb raw data, the following files have to be downloaded: title.basics.tsv.gz, title.crew.tsv.gz
* Here the documentation and dataset for for academic purposes only can be found: https://developer.imdb.com/non-commercial-datasets/


### Executing program

* The documentation of the preprocessing and dataset creation for IMDb can be found in: `data/IMDB/imdb.ipynb`
* The documentation for the evaluation and creation of the survey of IMDb can be found in: `data/IMDB/plots_imdb.ipynb'
* The implementation to create the datasets can found in `/scripts/emb_name_of_dataset.py`
* The output of the embeddings is saved under `data/dataset_name/embeddings/`
* To run the RGCN and RGAT with the different embedding types you have to run the `main.py`
* Here, you can adapt in the marked area the path, as well as the dataset, model and embedding types you want to train
* The IMDB dataset is only trainable on 'RGCN_no_emb' and 'RGAT_no_emb'
* Thereby, 'RGAT_emb' and 'RGCN_emb' in combination with the in the 'embs' parameter selected embeddings will be trained
* All models and variant are selectable by a list, such that the training of all models is carried out directly one after the other.
* After the training of one of these model is finished, the computation of LRP starts directly after.
* The model output is saved in the following folder: òut/dataset_name/model_name/(embedding_name/)relevances/`
* This structure has to exist, to be able to save the output files


* The implementation of the RGAT layer can be found in `/model/rgat_act.py`
* The implementation of the RGCN layer can be found in `/model/rgcn_layers.py`
* The implementation of the LRP algorithm can be found in `/model/lrp_act.py`



## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
