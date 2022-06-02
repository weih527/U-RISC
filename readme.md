# U-RISC: Ultra-high Resolution EM Images Segmentation Challenge

Organizer's paper: [U-RISC: An Annotated Ultra-High-Resolution Electron Microscopy Dataset Challenging the Existing Deep Learning Algorithms](https://www.frontiersin.org/articles/10.3389/fncom.2022.842760/full)

Official website: [Ultra-high Resolution EM Images Segmentation Challenge](https://www.biendata.xyz/competition/urisc/)

Our solution (**4th**): [U-RISC 电镜图像神经元分割 第四名解决方案](https://blog.csdn.net/qq_33757398/article/details/104421986?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522165415437516781432996346%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=165415437516781432996346&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-104421986-null-null.nonecase&utm_term=%E7%AC%AC%E5%9B%9B%E5%90%8D&spm=1018.2226.3001.4450)



## Simple Track

Preliminary leaderboard: **5/85 (VIDAR)**

![](./images/simple_pre_leaderboard.png)



Final leaderboard: **4th (VIDAR)**

![](./images/simple_final_leaderboard.png)



## Complex Track

Preliminary leaderboard: **3/36 (VIDAR)**

![](./images/complex_pre_leaderboard.png)



Final leaderboard: **4th (VIDAR)**

![](./images/complex_final_leaderboard.png)



## Paper results

![](./images/paper_results.png)



## Data

Download official data from [BaiduYun](https://pan.baidu.com/s/1SNNSMAvIi1KjqydHA6kkgg) (Access code: weih)



## Models

Download our trained models from [BaiduYun](https://pan.baidu.com/s/1FvsL_OXkpINPHNfFL9lWEA) (Access code: weih)

And then unzip the 'models5.zip' file



## Installation

If you have a [Docker](https://www.docker.com/) environment, we strongly recommend you to pull our image as follows,

```shell
docker pull renwu527/auto-emseg:v4.1
```



## Run

```shell
python run.py -in=/DATA-PATH
```

