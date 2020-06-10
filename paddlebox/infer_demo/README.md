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

Step1. Prepare `slot.txt` and the `model` directory. In this demo, the model description file is `model/program.bin` and the model parameter file is `model/model.bin` 

Step2. Run the demo application and a fetch list separated by comma could be set
```shell
sh run_demo.sh fetch_var1,fetch_var2,...
```
