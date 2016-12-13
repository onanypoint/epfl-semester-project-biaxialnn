import getopt, sys
import optparse
from argparse import ArgumentParser
import os 
import sys
import traceback
import pickle
import music21 as m21
import numpy as np
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    sp = parser.add_subparsers()
    
    preprocess = sp.add_parser("preprocess")
    preprocess.set_defaults(command="preprocess")
    preprocess.add_argument("-d", "--input-directory", help="Directory containing scores to transform to statematrix format.")
    preprocess.add_argument("-o", "--output-file", default="data/music_statematrix.pkl", help="Save statematrix pickle to this location.")
    preprocess.add_argument("-c", "--count", default=None, help="Number of randomly selected scores to process.")

    train = sp.add_parser("train")
    train.set_defaults(command="train")
    train.add_argument("-f", "--statematrix-file", default="data/music_statematrix.pkl", help="File containing statematrix pickel, i.e output of the preprocessing.")
    train.add_argument("-o", "--output-directory", default="output/", help="Where to save meta information during training.")
    train.add_argument("-e", "--training-epochs", type=int, default=10000, help="Number of iterations to run for.")
    train.add_argument("-t", "--t-layer-sizes", default=[300,300], help="List of size for each LSTM layer used for the time model.")
    train.add_argument("-p", "--p-layer-sizes", default=[100,50], help="List of size for each LSTM layer used for the pitch model.")
    train.add_argument("-r", "--dropout", type=float, default=0.5, help="Dropout value")
    train.add_argument("-v", "--validation-split", type=float, default=0.1, help="Percentage of pieces to keep for validation purposes.")
    train.add_argument("-m", "--model-config", default=None, help="Model config (trained weights) to load before starting training.")

    generate = sp.add_parser("generate")
    generate.set_defaults(command="generate")
    generate.add_argument("-t", "--t-layer-sizes", default=[300,300], help="List of size for each LSTM layer used for the time model.")
    generate.add_argument("-p", "--p-layer-sizes", default=[100,50], help="List of size for each LSTM layer used for the pitch model.")
    generate.add_argument("-r", "--dropout", type=float, default=0.5, help="Dropout value")
    generate.add_argument("-s", "--seed",  required=True, help="Score to use as seed for generation.")
    generate.add_argument("-l", "--length", type=int, default=80, help="Number of timestep to generate.")
    generate.add_argument("-c", "--conservativity", type=float, default=1, help="Conservativity value, i.e. how much freedom is given to the generation process.")
    generate.add_argument("-m", "--model-config", default="output/weights/params_final.p", help="Model config (trained weights) to load before starting generation.")
    generate.add_argument("-o", "--output-directory", default="output/", help="Where to save the generated samples.")
    generate.add_argument("-n", "--name", default="generated", help="Name of the generated sample.")

    args = parser.parse_args()

    from utils.data import DataManager
    from utils.features import FeatureBuilderArticulations
    from utils.model import Model
    from utils.statematrix import StateMatrixBuilderArticulations
    from utils.training import train_piece

    datamanager = DataManager(FeatureBuilderArticulations(),StateMatrixBuilderArticulations())

    if args.command == 'preprocess':
        dirpath = args.input_directory
        files = [os.path.join(dirpath, fname) for fname in os.listdir(dirpath)]
        print("Processing {} pieces".format(len(files)))
        pieces = datamanager.pieces_to_statematrix(files, args.count)
        with open(args.output_file, 'wb') as o:
            pickle.dump(pieces, o)

    elif args.command == 'train':
        os.makedirs(args.output_directory, exist_ok=True)
        os.makedirs(args.output_directory + 'samples/', exist_ok=True)
        os.makedirs(args.output_directory + 'weights/', exist_ok=True)
       
        pieces = pickle.load(open(args.statematrix_file, 'rb'))
        model = Model(datamanager, args.t_layer_sizes, args.p_layer_sizes, dropout=args.dropout)
        model.setup()

        if args.model_config:
            model.learned_config = pickle.load(open(args.model_config,'rb'))
        
        print("Training")
        train_piece(model, pieces, args.training_epochs, directory=args.output_directory, validation_split=args.validation_split)
        
        print("Dumping")
        pickle.dump(model.learned_config, open( args.output_directory + 'weights/params_final.p', "wb" ) )

    elif args.command == 'generate':
        os.makedirs(args.output_directory, exist_ok=True)
        os.makedirs(args.output_directory + 'samples/', exist_ok=True)
        
        model = Model(datamanager, args.t_layer_sizes, args.p_layer_sizes, dropout=args.dropout)
        model.setup_generate()
        model.learned_config = pickle.load(open(args.model_config,'rb'))
        
        try:
            stream = m21.converter.parse(args.seed)
            seed_state = datamanager.s.stream_to_statematrix(stream)
            seed_feature = datamanager.f.note_state_matrix_to_input_form(seed_state)
            generated_sample = model.generate_fun(args.length, args.conservativity, seed_feature)
            statematrix = np.concatenate((seed_state, generated_sample), axis=0)
            s = model.data_manager.s.statematrix_to_stream(statematrix)
            np.save(args.output_directory + 'samples/{}.npy'.format(args.name), statematrix)
            s.write('musicxml', args.output_directory + 'samples/{}.xml'.format(args.name))
            print("Done generating")
        except :
            print("Error, please try with another seed")

if __name__ == "__main__":
    main()