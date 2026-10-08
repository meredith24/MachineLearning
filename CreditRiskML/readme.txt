Bu projede kullanılan veri seti kaggle'dan (https://www.kaggle.com/datasets/laotse/credit-risk-dataset) alınmıştır.

XGBoost Classifier ve Linear Regression ile modellenme yapılmıştır. Her ne kadar açıklanabilirliği ve yorumlanabilirliği sınırlı olsa da
başarı metriklerine bakıldığında XGBoost Classifier'ın üstünlüğü gözlemlenmiştir. Sklearn modülünden TunedThresholdClassifierCV sınıfı kulllanılarak modelin başarı metriklerinden
recall(hassasiyet) ve precision(kesinlik) arasındaki denge sağlanmaya çalışılmıştır. Bunun için örnekleri sınıflandırma eşik değeri "0.24" seçilmiştir.

Modeli dışarı çıkarmak için Backend tarafında Fastapi ile bir yapı oluşturulmuştur.

Frontend yapısını oluşturmak için yapay zeka araçlarından(Cursor) destek alınmıştır.

Program için proje kökündeki api klasörü içerinden api_endpoints.py isimli proje çalıştırılır.
Ardından terminal kısmına "uvicorn api.api_endpoints:app --reload" yazılarak kullanıcı arayüzüne ulaşılır.