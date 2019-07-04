# image_match
locate small image in big image

example：
python3 main.py tanker.py tanker_a.png
python3 main.py tanker.py tanker_b.jpg

使用example时请注意：
在检测绿色飞行器时， threshold_score = 0.9, threshold_dist = 0.4
在检测棕色飞行器时， threshold_score = 0.8, threshold_dist = 0.6

如果使用其他图片测试， 需要调试threshold_dist，
