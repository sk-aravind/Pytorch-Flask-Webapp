import pandas as pd

def replace_x(x):
    x = x.replace("..", "/static")
    return x

def clip_dec(x):
    return float("{0:.3f}".format(x))


df = pd.read_csv('scores_wth_gt-1.csv')
frames = []

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

for cat in object_categories:
    top50df = df.sort_values(by=cat, ascending=False).head(50)
    frames.append(top50df)

result_df = pd.concat(frames)
result_df['image'] = result_df['image'].apply(replace_x)

for cat in object_categories:
    result_df[cat] = result_df[cat].apply(clip_dec)

result_df.to_csv("results-20-combined-5.csv", index = False)