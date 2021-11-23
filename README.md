# HyperTransformer
Official PyTorch implementation of the paper: HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening

# Download datasets

We use three publically available HSI datasets for experiments, namely

1) Pavia Center scene [Download the .mat file here](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes), and save it in "./datasets/pavia_centre/Pavia_centre.mat".
2) Botswana [Download the .mat file here](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes), and save it in "./datasets/botswana4/Botswana.mat".
3) Chikusei datasets [Download the .mat file here](https://naotoyokoya.com/Download.html), and save it in "./datasets/chikusei/chikusei.mat".


 # Processing the datasets to generate LR-HSI, PAN, and Reference-HR-HSI using Wald's protocol
 We use Wald's protocol to generate LR-HSI and PAN image. To generate those cubic patches,
  1) Run `process_pavia.m` in `./datasets/pavia_centre/` to generate cubic patches. 
  2) Run `process_botswana.m` in `./datasets/botswana4/` to generate cubic patches.
  3) Run `process_chikusei.m` in `./datasets/chikusei/` to generate cubic patches.
 
# Training HyperTransformer 
We use two stage procedure to train our HyperTransformer. We first train HyperTransformer without proposed MHFA and then fine-tune MHFA.

## Training the HyperTransformer without MHFA
Use the following codes to pre-train HyperTransformer on the three datasets.
 1) Pre-training on Pavia Center Dataset: 
    
    Change "train_dataset" to "pavia_dataset" in config_HSIT_PRE.json. 
    
    Then use following commad to pre-train on Pavia Center dataset.
    `python train.py --config "configs/config_HSIT_PRE.json"`.
    
 4) Pre-training on Botswana Dataset:
     Change "train_dataset" to "botswana4_dataset" in config_HSIT_PRE.json. 
     
     Then use following commad to pre-train on Pavia Center dataset. 
     `python train.py --config "configs/config_HSIT_PRE.json"`.
     
 6) Pre-training on Chikusei Dataset: 
     
     Change "train_dataset" to "chikusei_dataset" in config_HSIT_PRE.json. 
     
     Then use following commad to pre-train on Pavia Center dataset. 
     `python train.py --config "configs/config_HSIT_PRE.json"`.
     

## Fine tuining the MHFA in HyperTransformer
Next, we fine-tune the MHFA module in HyperTransformer starting from pre-trained weights we obtained in the previous step.
 1) Fine-tuning MHFA on Pavia Center Dataset: 

    Change "train_dataset" to "pavia_dataset" in config_HSIT.json. 
    
    Then use the following commad to train HyperTransformer on Pavia Center dataset. 
    
    Please specify path to best model obtained from previous step using --resume.
    `python train.py --config configs/config_HSIT.json --resume ./Experiments/HSIT_PRE/pavia_dataset/N_modules\(4\)/best_model.pth" `.
   
 3) Fine-tuning on Botswana Dataset: 

    Change "train_dataset" to "botswana4_dataset" in config_HSIT.json. 
    
    Then use following commad to pre-train on Pavia Center dataset. 
    
    `python train.py --config configs/config_HSIT.json --resume ./Experiments/HSIT_PRE/botswana4/N_modules\(4\)/best_model.pth`.

 5) Fine-tuning on Chikusei Dataset: 

    Change "train_dataset" to "chikusei_dataset" in config_HSIT.json.
    
    Then use following commad to pre-train on Pavia Center dataset. 
    
    `python train.py --config configs/config_HSIT.json --resume ./Experiments/HSIT_PRE/chikusei_dataset/N_modules\(4\)/best_model.pth`.



