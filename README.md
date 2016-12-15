# Music Score Generation

## Getting Started

These instructions will get you a copy of the project up and running on your machine.

### Prerequisites

The codebase is written in python. The bare minimum you should do to get everything running, assuming you have Python and conda installed, is

```shell
# Clone the repo
git clone git://XXXXXXXXXX biaxialnn
cd biaxialnn

# Install dependencies
conda create -n biaxialnn python=3.5
source activate biaxialnn
pip install -r requirements.txt
```

If you want to run the model on GPU, you also need to have a _```theanorc```_ ([more info](http://deeplearning.net/software/theano/library/config.html)) with GPU enable. You might also want to look at this [page](http://deeplearning.net/software/theano/tutorial/using_gpu.html) related to CUDA backend.

### Before starting

Before running any machine learning code, you will have to obtain a large collection of score file. Those should be placed in a data folder dedicated to "raw" scores.

There is a [jupyter notebook]() included showing an example of a pipeline going through the data retrieval from [musescore.com](http://musescore.com/). The pre-processing to go from the "raw" score to the statematrix representation. And finally the training of the model before generating a score.

Other source of "raw" scores:
- [piano-midi.de/](http://www.piano-midi.de/)
- [musescore.com](ttp://www.musescore.com)

## Interface

### Configuration file

The first step is to create a configuration file. There is an example file included.

```
cp config.ini.example confi.ini
```

Configuration options are:

```
pitch_lowerBound        #   Minimum midi pitch number taken into account during 
                            the processing phase. If a note lower than that is 
                            encountered, it will be discarded and a warning 
                            will be shown.

pitch_upperBound        #   Maximum midi pitch number taken into account.

measure_quantization    #   In how many timestep a measure is divided into. By
                            default it is 16 meaning that each quarter note is
                            "subdivided" into four. Making the 16th note the 
                            shortest note which can be represented.

batch_size              #   Number of score segment to process in parallel
                            during the training phase. Be aware that the bigger
                            the number is, the greater chance of an "out of 
                            memory" error.

seq_len                 #   Length of the sequence used during training.       

division_len            #   Minimum number of timestep between two sequences 
                            taken from the same score.

musescore_api_key       #   Musescore API Key. Only used during data retrieval.
                            More info can be found on developers.musescore.com
```
A CLI implementation is included.

### Usage


```shell
$ python main.py [-h] {preprocess, train, generate}
```

Without any options running the preprocess, train and generate command will be
run using the default values and on the minimal dataset present in the repository.

##### Warning
A training iteration takes around 10 seconds on a GTX 650. The training can be
gracefully stopped with ```CTRL-C```.

-------------

```shell
$ python main.py preprocess --help
usage: main.py preprocess [-h] [-d INPUT_DIRECTORY] [-o OUTPUT_FILE]
                          [-c COUNT]

optional arguments:
  -h, --help            show this help message and exit
  -d INPUT_DIRECTORY, --input-directory INPUT_DIRECTORY
                        Directory containing scores to transform to
                        statematrix format.
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        Save statematrix pickle to this location.
  -c COUNT, --count COUNT
                        Number of randomly selected scores to process.
```


```shell
$ python main.py train --help                    
  -h, --help            show this help message and exit
  -f STATEMATRIX_FILE, --statematrix-file STATEMATRIX_FILE
                        File containing statematrix pickle, i.e output of the
                        preprocessing.
  -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                        Where to save meta information during training.
  -e TRAINING_EPOCHS, --training-epochs TRAINING_EPOCHS
                        Number of iterations to run for.
  -t T_LAYER_SIZES, --t-layer-sizes T_LAYER_SIZES
                        List of size for each LSTM layer used for the time
                        model.
  -p P_LAYER_SIZES, --p-layer-sizes P_LAYER_SIZES
                        List of size for each LSTM layer used for the pitch
                        model.
  -r DROPOUT, --dropout DROPOUT
                        Dropout value
  -v VALIDATION_SPLIT, --validation-split VALIDATION_SPLIT
                        Percentage of pieces to keep for validation purposes.
  -m MODEL_CONFIG, --model-config MODEL_CONFIG
                        Model config (trained weights) to load before starting
                        training.
```

```shell
$ python main.py generate --help
usage: main.py generate [-h] [-t T_LAYER_SIZES] [-p P_LAYER_SIZES]
                        [-r DROPOUT] -s SEED [-l LENGTH] [-c CONSERVATIVITY]
                        [-m MODEL_CONFIG] [-o OUTPUT_DIRECTORY] [-n NAME]

optional arguments:
  -h, --help            show this help message and exit
  -t T_LAYER_SIZES, --t-layer-sizes T_LAYER_SIZES
                        List of size for each LSTM layer used for the time
                        model.
  -p P_LAYER_SIZES, --p-layer-sizes P_LAYER_SIZES
                        List of size for each LSTM layer used for the pitch
                        model.
  -r DROPOUT, --dropout DROPOUT
                        Dropout value
  -s SEED, --seed SEED  Score to use as seed for generation.
  -l LENGTH, --length LENGTH
                        Number of timestep to generate.
  -c CONSERVATIVITY, --conservativity CONSERVATIVITY
                        Conservativity value, i.e. how much freedom is given
                        to the generation process.
  -m MODEL_CONFIG, --model-config MODEL_CONFIG
                        Model config (trained weights) to load before starting
                        generation.
  -o OUTPUT_DIRECTORY, --output-directory OUTPUT_DIRECTORY
                        Where to save the generated samples.
  -n NAME, --name NAME  Name of the generated sample.
```

## Built With

* [Theano](http://www.deeplearning.net/software/theano/) - Machine learning framework
* [Music21](http://web.mit.edu/music21/) - Toolkit for computer-aided musicology

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Daniel Johnson's [blogpost](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/) on music generation
* Karpathy RNN [blogpost](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)