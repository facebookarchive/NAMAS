# Attention-Based Summarization

This project contains the Abs. neural abstractive summarization system from the paper

     A Neural Attention Model for Abstractive Summarization.
     Alexander M. Rush, Sumit Chopra, Jason Weston.

The release includes code for:

* Extracting the summarization data set
* Training the neural summarization model
* Constructing evaluation sets with ROUGE
* Tuning extractive features

## Setup

To run the system, you will need to have [Torch7](http://torch.ch/)
installed. You will also need Python 2.7, NLTK, and
GNU Parallel to run the data processing scripts.  Additionally the
code currently requires a CUDA GPU for training and decoding.

Finally the scripts require that you set the $ABS environment variable.

    > export ABS=$PWD
    > export LUA_PATH="$LUA_PATH;$ABS/?.lua"

## Constructing the Data Set

The model is trained to perform title generation from the first line
of newspaper articles. Since the system is completely data-driven it
requires a large set of aligned input-title pairs for training.

To provide these pairs we use the [Annotated Gigaword
corpus](https://catalog.ldc.upenn.edu/LDC2012T21) as our main data
set. The corpus is available on LDC, but it requires membership.  Once
the annotated gigaword is obtained, you can simply run the provided
script to extract the data set in text format.

### Generating the data

To construct the data set run the following script to produce `working_dir/`,
where `working_dir/' is the path to the directory where you want to store the
processed data. The script 'construct_data.sh' makes use of the 'parallel'
utility, so please make sure that it is in your path.
WARNING: This may take a couple hours to run.

     > ./construct_data.sh agiga/ working_dir/

### Format of the data files

The above command builds aligned files of the form split.type.txt where split
is train/valid/test and type is title/article.

The output of the script is several aligned plain-text files.
Each has one title or article per line.

     > head train.title.txt
     australian current account deficit narrows sharply
     at least two dead in southern philippines blast
     australian stocks close down #.# percent
     envoy urges north korea to restart nuclear disablement
     skorea announces tax cuts to stimulate economy

These files can be used to train the ABS system or be used by other baseline models.

## Training the Model

Once the data set has been constructed, we provide a simple script to train
the model.

   > ./train_model.sh working_dir/ model.th


The training process consists of two stages. First we convert the text
files into generic input-title matrices and then we train a
conditional NNLM on this representation.

Once the model has been fully trained (this may require 3-4 days),
you can use the test script to produce summaries of any plain text file.w

   > ./test_model.sh working_dir/valid.article.filter.txt model.th length_of_summary


### Training options

These scripts utilize the Torch code available in `$ABS/summary/`

There are two main torch entry points. One for training the model
from data matrices and the other for evaluating the model on plain-text.

     > th summary/train.lua -help

     Train a summarization model.

       -articleDir      Directory containing article training matrices. []
       -titleDir        Directory containing title training matrices. []
       -validArticleDir Directory containing article matricess for validation. []
       -validTitleDir   Directory containing title matrices for validation. []
       -auxModel        The encoder model to use. [bow]
       -bowDim          Article embedding size. [50]
       -attenPool       Attention model pooling size. [5]
       -hiddenUnits     Conv net encoder hidden units. [1000]
       -kernelWidth     Conv net encoder kernel width. [5]
       -epochs          Number of epochs to train. [5]
       -miniBatchSize   Size of training minibatch. [64]
       -printEvery      How often to print during training. [1000]
       -modelFilename   File for saving loading/model. []
       -window          Size of NNLM window. [5]
       -embeddingDim    Size of NNLM embeddings. [50]
       -hiddenSize      Size of NNLM hidden layer. [100]
       -learningRate    SGD learning rate. [0.1]



### Testing options


The run script is used for beam-search decoding with a trained
model. See the paper for a description of the extractive
features used at decoding time.

    > th summary/run.lua -help

    -blockRepeatWords Disallow generating a repeated word. [false]
    -allowUNK         Allow generating <unk>. [false]
    -fixedLength      Produce exactly -length words. [true]
    -lmWeight         Weight for main model. [1]
    -beamSize         Size of the beam. [100]
    -extractive       Force fully extractive summary. [false]
    -lmWeight         Feature weight for the neural model. [1]
    -unigramBonus     Feature weight for unigram extraction. [0]
    -bigramBonus      Feature weight for bigram extraction. [0]
    -trigramBonus     Feature weight for trigram extraction. [0]
    -lengthBonus      Feature weight for length. [0]
    -unorderBonus     Feature weight for out-of-order extraction. [0]
    -modelFilename    Model to test. []
    -inputf           Input article files.  []
    -nbest            Write out the nbest list in ZMert format. [false]
    -length           Maximum length of summary.. [5]



## Evaluation Data Sets

We evaluate the ABS model using the shared task from the Document Understanding Conference (DUC).

This release also includes code for interactive with the DUC shared
task on headline generation. The scripts for processing and evaluating
on this data set are in the DUC/ directory.

The [DUC data set](http://duc.nist.gov/duc2004/tasks.html) is
available online, unfortunately you must manually fill out a form to
request the data from NIST.  Send the request to
[Angela Ellis](mailto:angela.ellis@nist.gov).

### Processing DUC

After receiving credentials you should obtain a series of
tar files containing the data used as part of this shared task.

1. Make a directory DUC_data/ which should contain the given files


       >DUC2003\_Summarization\_Documents.tgz
       >DUC2004\_Summarization\_Documents.tgz
       >duc2004\_results.tgz
       >detagged.duc2003.abstracts.tar.gz

2. Run the setup script (this requires python and NLTK for tokenization)


      > ./DUC/setup.sh DUC_data/


After running the scripts there should be directories

       DUC_data/clean_2003/
       DUC_data/clean_2004/


Each contains a file input.txt where each line is a tokenized first line of an article.


     > head DUC_data/clean_2003/input.txt
     schizophrenia patients whose medication could n't stop the imaginary voices in their heads gained some relief after researchers repeatedly sent a magnetic field into a small area of their brains .
     scientists trying to fathom the mystery of schizophrenia say they have found the strongest evidence to date that the disabling psychiatric disorder is caused by gene abnormalities , according to a researcher at two state universities .
     a yale school of medicine study is expanding upon what scientists know  about the link between schizophrenia and nicotine addiction .
     exploring chaos in a search for order , scientists who study the reality-shattering mental disease schizophrenia are becoming fascinated by the chemical environment of areas of the brain where perception is regulated .

As well as a set of references:


    > head DUC_data/clean_2003/references/task1_ref0.txt
    Magnetic treatment may ease or lessen occurrence of schizophrenic voices.
    Evidence shows schizophrenia caused by gene abnormalities of Chromosome 1.
    Researchers examining evidence of link between schizophrenia and nicotine addiction.
    Scientists focusing on chemical environment of brain to understand schizophrenia.
    Schizophrenia study shows disparity between what's known and what's provided to patients.

System output should be added to the directory system/task1_{name}.txt. For instance the script includes a baseline PREFIX system.


    DUC_data/clean_2003/references/task1_prefix.txt


### ROUGE for Eval

To evaluate the summaries you will need the [ROUGE eval system](http://research.microsoft.com/~cyl/download/ROUGE-1.5.5.tgz).

The ROUGE script requires output in a very complex HTML form.
To simplify this process we include a script to convert the
simple output to one that ROUGE can handle.

Export the ROUGE directory `export ROUGE={path_to_rouge}` and then run the eval scripts


    > ./DUC/eval.sh DUC_data/clean_2003/
    FULL LENGTH
       ---------------------------------------------
       prefix ROUGE-1 Average_R: 0.17831 (95%-conf.int. 0.16916 - 0.18736)
       prefix ROUGE-1 Average_P: 0.15445 (95%-conf.int. 0.14683 - 0.16220)
       prefix ROUGE-1 Average_F: 0.16482 (95%-conf.int. 0.15662 - 0.17318)
       ---------------------------------------------
       prefix ROUGE-2 Average_R: 0.04936 (95%-conf.int. 0.04420 - 0.05452)
       prefix ROUGE-2 Average_P: 0.04257 (95%-conf.int. 0.03794 - 0.04710)
       prefix ROUGE-2 Average_F: 0.04550 (95%-conf.int. 0.04060 - 0.05026)


## Tuning Feature Weights

For our system ABS+ we additionally tune extractive features on the DUC
summarization data. The final features we obtained our distributed with the
system as `tuning/params.best.txt`.

The MERT tuning code itself is located in the `tuning/` directory. Our setup
uses [ZMert](http://cs.jhu.edu/~ozaidan/zmert/) for this process.

It should be straightforward to tune the system on any developments
summarization data. Take the following steps to run tuning on the
DUC-2003 data set described above.

First copy over reference files to the tuning directoy. For instance to tune on DUC-2003:

    ln -s DUC_data/clean_2003/references/task1_ref0.txt tuning/ref.0
    ln -s DUC_data/clean_2003/references/task1_ref1.txt tuning/ref.1
    ln -s DUC_data/clean_2003/references/task1_ref2.txt tuning/ref.2
    ln -s DUC_data/clean_2003/references/task1_ref3.txt tuning/ref.3

Next copy the SDecoder template, `cp SDecoder_cmd.tpl SDecoder_cmd.py`
and modify the `SDecoder_cmd.py` to point to the model and input text.

    {"model" : "model.th",
     "src" : "/data/users/sashar/DUC_data/clean_2003/input.txt",
     "title_len" : 14}


Now you should be able to run Z-MERT and let it do its thing.

    > cd tuning/; java -cp zmert/lib/zmert.jar ZMERT ZMERT_cfg.txt

When Z-MERT has finished you can run on new data using command:

    > python SDecoder_test.py input.txt model.th
