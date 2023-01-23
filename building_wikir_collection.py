import argparse
import wikIR_Collection_opt
import time


################2. Building Wikir Collections####################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--coll_path', nargs="?", type=str)
    parser.add_argument('-i', '--indexed_path', nargs="?", type=str)
    parser.add_argument('-f', '--fasttext_path', nargs="?", type=str)

    args = parser.parse_args()

    print(args, flush=True)
    print("------------------start---------------",flush=True)
    print('--------Reading documents', flush=True)
    start=time.time()
    Collection_wikir = wikIR_Collection_opt.Collection('english')
    Collection_wikir.load_documents(args.coll_path)
    
    print('------Creating Posting file, Vocabulary and Document length and filtering tokens ', flush=True)
    Collection_wikir.build_inverted_index_and_vocabulary(args.indexed_path,minimum_occurence=5,proportion_of_frequent_words=0.2)

#     print('Compute fasstetext embeddings', flush=True)
#     Collection_wikir.compute_fasttext_embedding(args.fasttext_path)
    end=time.time()
    print('--------------------Finished------------------',flush=True)
    print("Total time = ", round(end-start), " s",flush=True)
if __name__ == "__main__":
    main()
