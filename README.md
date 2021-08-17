# relation_extraction-baseline
# 所需环境
Python==3.6</br>
tensorflow==1.14.0</br>
keras==2.3.1</br>
bert4keras==0.10.6</br>
# 项目目录
├── bert4keras</br>
├── data &emsp; &emsp;存放数据</br>
├── pretrained_model &emsp; &emsp;存放预训练模型</br>
├── save &emsp; &emsp;存放已训练好的模型</br>
├── relation_extraction_train.py &emsp; &emsp;训练代码</br>
├── relation_extraction_predict.py &emsp; &emsp;评估和测试代码</br>
# 数据集
[2019语言与智能技术竞赛信息抽取任务](https://ai.baidu.com/broad/download?dataset=)</br>
[NYT](https://drive.google.com/open?id=10f24s9gM7NdyO3z5OqQxJgYud4NnCJg3)</br>
# 使用说明
1.[下载预训练语言模型](https://github.com/google-research/bert#pre-trained-models)</br>
&emsp; &emsp; 中文数据集采用BERT-Base, Chinese</br>
&emsp; &emsp; 英文数据集采用BERT-Base, Cased</br>
2.构建数据集</br>
&emsp; &emsp;中文：将下载的信息抽取任务数据集放到data/baidu/文件夹下</br>
&emsp; &emsp;英文：将下载的NYT数据集放到data/NYT/raw_NYT/文件夹下，</br>
&emsp; &emsp;&emsp;&emsp;&emsp;运行generate.py生成train_data.json、valid_data.json和test_data.json</br>
3.训练模型
```
python relation_extraction_train.py
```
4.评估和测试
```
python relation_extraction_predict.py
```
# 模型展示
三元组模型展示参考项目[relation_extraction-Demo](https://github.com/dreams-flying/relation_extraction-Demo)
![image](https://github.com/dreams-flying/relation_extraction-baseline/blob/master/images/demo.png)
