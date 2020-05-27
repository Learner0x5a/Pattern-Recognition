# COD数据集图像分类

## 数据预处理

 + `crop.py`: 读取坐标、标签、裁剪图片、统一大小为 (144, 144)

## PCA + KNN

 + `pca_knn.py`: GLASS/MIRROR二分类
 + `openset_pca_knn.py`: 开集测试

## CNN

 + `cnn.py`: GLASS/MIRROR二分类
 + `openset-cnn.py`: 开集测试
 + `closeset-cnn.py`: 把OTHERS加入训练集