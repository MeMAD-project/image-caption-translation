Installing
==========

Download model
--------------

    curl https://zenodo.org/record/4038444/files/opennmt.transformer.multiling.mscoco%2Bmulti30k%2Bsubs3M.domainprefix.mmod.imgw.meanfeat.detectron.mask_surface.bpe50k_acc_80.57_ppl_2.43_e23.pt?download=1 \
    --output models/opennmt.transformer.multiling.mscoco+multi30k+subs3M.domainprefix.mmod.imgw.meanfeat.detectron.mask_surface.bpe50k_acc_80.57_ppl_2.43_e23.pt

Image feature extraction
------------------------

This codebase uses CUDA by default but can be also used with a CPU.

    conda create --name memaddetectron2 --file env/detectron2.cuda.conda -c pytorch
    source activate memaddetectron2
    pip install -r env/detectron2.cuda.pip
	source deactivate

Translation system
------------------

Note that this codebase uses old versions of libraries. CUDA is disabled.

    conda create --name memadmmt --file env/mmt.nocuda.conda -c pytorch
    source activate memadmmt
    pip install -r env/mmt.nocuda.pip
    git clone https://github.com/Waino/OpenNMT-py.git
    pushd OpenNMT-py
    git checkout develop_mmod
    python setup.py install
    popd

Usage
=====

Image feature extraction
------------------------

Extract detectron2 features and store them in `img_feat.npy`.

    source activate memaddetectron2
	tools/image-features.py --imglist data/imglist
	source deactivate

Note: `img_feat_dim=80, dtype=torch.float32`.
Saved as a (N,80) matrix in numpy `.npy` format, where N is the number of lines to translate.

Translation system
------------------

Apply BPE segmentation to tokenized, lowercased text.

    source activate memadmmt
    tools/apply_bpe.py --codes models/bpe.50k.multiling < data/input > data/segmented

Preprend the target language tag (either `TO_de` or `TO_fr`) and the domain tag.

    sed -e "s/^/<TO_de> <DOMAIN_caption> /" < data/segmented > data/prefixed.de
    sed -e "s/^/<TO_fr> <DOMAIN_caption> /" < data/segmented > data/prefixed.fr

Perform translation.
(This example feeds in a zero vector as dummy features. This has less effect on the result than you might expect.)

    OpenNMT-py/translate_mmod_finetune.py \
        -model models/opennmt.transformer.multiling.mscoco+multi30k+subs3M.domainprefix.mmod.imgw.meanfeat.detectron.mask_surface.bpe50k_acc_80.57_ppl_2.43_e23.pt \
        -src data/prefixed.de \
        -path_to_test_img_feats dummy.zeros.npy \
        -output data/translated.de \
        --multimodal_model_type imgw

Postprocess translation: join BPE subwords, recase.

Citing
======

If you use this model in a scientific publication, please cite

    @inproceedings{gronroos2018memad,
        title = {The {MeMAD} Submission to the {WMT18} Multimodal Translation Task},
        author = {Gr{\"o}nroos, Stig-Arne and Huet, Benoit and Kurimo, Mikko and Laaksonen, Jorma and Merialdo, Bernard and Pham, Phu and Sj{\"o}berg, Mats and Sulubacak, Umut and Tiedemann, J{\"o}rg and Troncy, Raphael and V{\'a}zquez, Ra{\'u}l},
        year = {2018},
        month = {October},
        booktitle = {Proceedings of the Third Conference on Machine Translation},
        publisher = {Association for Computational Linguistics},
        url = {http://www.aclweb.org/anthology/W18-6439.pdf}
    }

