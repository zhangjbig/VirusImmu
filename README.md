# VirusImmu


## Installation
System Requirements
Python = 3.7
pandas = 1.1.4
pickle = 4.0
sklearn = 0.24.1
xgboost = 1.2.0

## Start
`pip install pandas`
`pip install pickle`
`pip install sklearn`
`pip install xgboost`
`pip install random`

## How to run

### Parameters
    descriptor = E
    L = 8
    Thred = 0.4
    Model_name = 'VirusImmu'

### Use the E-descriptor to represent protein sequences
        Model.descriptor_made_txt(content)
        input: Protein sequences in text format(content)
        output: the results represented by the E-descriptor, the output address is：./data/test/descriptor_result/result.csv


### ACC transformation
        Model.ACC_caculated()
        output: the result of ACC transformation, the output address is ./data/test/acc_result/E_ACC_l=8.csv


### immunogenicity prediction by our model
        python Run.py main()
        output: the predicted result of the 'xxxx' model for the input protein sequence, the output address is: ./output/result.csv
