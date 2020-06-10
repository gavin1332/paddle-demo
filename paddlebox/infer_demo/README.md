# HOW TO USE

## Run demo

Step1. Prepare environment
```shell
sh prepare_env.sh 
```

Step2. Compile source code of demo and run with default settings
```shell
sh compile_demo_and_run.sh 
```

Step3. Run the demo application and a fetch list separated by comma could be set
```shell
sh run_demo.sh fc_1.tmp_1,fc_0.tmp_2

```

## Predict with real model

Step1. Prepare environment
```shell
sh prepare_env.sh 
```

Step2. Compile source code of demo and run with default settings
```shell
sh compile_demo_and_run.sh 
```

Step3. Prepare `slot.txt` and the `model` directory. The model description file is `model/program.bin` and the model parameter file is `model/model.bin` 

Step4. Prepare input data in `data/sample.data` in the following format, you could refer to the data format of the demo above
```
slot_name_1\tf11 f12 ... f1n
slot_name_2\tf21 f22 ... f2n
...
slot_name_m\tfm1 fm2 ... fmn
```
The slot name and the feature section is separated by a tab and all dense features are separated by space.

NOTICE: `slot_name_xxx` should be consistant to slots configured in `slot.txt`

Step5. Run the demo application and a fetch list separated by comma could be set
```shell
sh run_demo.sh fetch_var1,fetch_var2,...
```
