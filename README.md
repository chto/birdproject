# Hand Signs Recognition with Tensorflow

*Authors: Olivier Moindrot and Guillaume Genthial*


## Task

Identify the species of the birds from their sound.


## Download the SIGNS dataset

Here is the structure of the data:
```
data/
    train/
        bird_species1/
            sound_1.wmv
            sound_2.wmv
            ...
    valid/
        bird_species1/
            sound_1.wmv
            sound_2.wmv
            ...
        ...


```


## Quickstart (~30 min)

1. __First experiment__ We created a `base_model` directory for you under the `experiments` directory. It countains a file `params.json` which sets the parameters for the experiment. It looks like
```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 10,
    ...
}
```
For every new experiment, you will need to create a new directory under `experiments` with a similar `params.json` file.

3. __Train__ your experiment. Simply run
```
python train.py --data_dir data/ --model_dir experiments/base_model --noise_dir noisedir
```

[SIGNS]: https://drive.google.com/file/d/1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC/view?usp=sharing
