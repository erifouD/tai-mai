import pandas as pd

def get_type(model_name):
    name = str(model_name).upper()
    
    fighters = ['F-', 'F/A', 'TORNADO', 'EUROFIGHTER', 'HAWK', 'SPITFIRE', 'SR-71']
    for f in fighters:
        if f in name:
            return 1
            
    if 'C-130' in name:
        return 2
        
    business = ['CESSNA', 'FALCON', 'GULFSTREAM', 'BEECHCRAFT', 'PA-28', 'CHALLENGER', 'GLOBAL', 'KING AIR', 'DR-400', 'SR20']
    for b in business:
        if b in name:
            return 3
            
    return 0

images_dir = 'dataset/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data/images/'
csv_files = ['dataset/train.csv', 'dataset/val.csv', 'dataset/test.csv']

data = []

for csv in csv_files:
    df = pd.read_csv(csv)
    for i, row in df.iterrows():
        img_path = images_dir + row['filename']
        model = row['Classes']
        t = get_type(model)
        
        data.append({
            'image_path': img_path,
            'model_label': model,
            'type_label': t
        })

res = pd.DataFrame(data)
res.to_csv('dataset_metadata.csv', index=False)
print("Скрипт отработал, таблица сохранена! Всего картинок:", len(res))
