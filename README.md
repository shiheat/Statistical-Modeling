# Statistical-Modeling
Statistical Modeling


## 需求：
根据设备的时间标签、位置标签、红外传感器数据等信息，针对人员流动进行预测。（例如：再某段时间内，某个地点A的红外检测信号为0，则代表没有人，另外一段时间内的红外检测信号为1，则代表有人。在此期间，有其他的地点B的红外信息从1变成0，则代表人员从B-->A ）

## 需要的结果
形成人员流动的有向图（比如说，在早上最开始只有某地点A红外检测为1后面不停变换到其他的地点，从早到晚可以形成一个人员流动的有向图A-->B-->C...）

节点(Node):每个节点表示建筑物中的一个房间或区域。
边(Edge):每条边表示连接两个房间或区域的路径。
权重(Weight):边的权重表示通过该路径的人流量。
方向(Direction):边的方向表示人流的流向。


## 之前他们做的是利用下面这个项目改的。
https://github.com/VeritasYin/STGCN_IJCAI-18/blob/master/README.md

https://www.ijcai.org/proceedings/2018/0505.pdf

