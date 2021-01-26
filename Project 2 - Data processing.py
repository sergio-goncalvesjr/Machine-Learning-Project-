#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
uri = 'https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv'
dados = pd.read_csv(uri)


# In[25]:


dados.head()


# In[26]:


a_renomear = {
    'expected_hours' : 'horas_esperadas',
    'price' : 'preco',
    'unfinished' : 'nao_finalizado'
    
}
dados = dados.rename(columns = a_renomear)
dados.head()


# In[27]:


troca = {
    0 : 1,
    1 : 0
}
dados['finalizado'] = dados.nao_finalizado.map(troca)
dados.head()


# In[28]:


dados.tail()


# In[29]:


import seaborn as sns

sns.scatterplot(x='horas_esperadas', y = 'preco', data = dados)


# In[30]:


sns.scatterplot(x='horas_esperadas', y = 'preco',hue = 'finalizado', data = dados)


# In[31]:


sns.relplot(x = 'horas_esperadas', y = 'preco', hue = 'finalizado',col = 'finalizado', data = dados)


# In[32]:


x = dados[['horas_esperadas', 'preco']]
y = dados['finalizado']


# In[36]:


import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SEED = 5
np.random.seed(SEED)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, 
                                                        test_size = 0.25,
                                                        stratify = y)
print("Treinaremos com %d elementos e testaremos %d elementos" % (len(treino_x), len(teste_x)))


modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print('A acurácia é de %.2f%%' % acuracia)


# In[ ]:


# previsoes de base refletem a porcentagem minima de acerto
import numpy as np

previsoes_baseline = np.ones(540)

acuracia = accuracy_score(teste_y, previsoes_baseline) * 100
print('A acurácia é de %.2f%%' % acuracia)


# In[ ]:


sns.scatterplot(x='horas_esperadas', y = 'preco',hue = teste_y , data = teste_x)


# In[ ]:


# Referente ao máximo da linha x que vai de 0 a 100 e de y que vai de 101 a 27738
x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()
print(x_min, x_max, y_min, y_max)


# In[ ]:


pixels = 100
eixo_x = np.arange (x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange (y_min, y_max, (y_max - y_min) / pixels)


# In[ ]:


#agora iremos criar um grid com os eixos x e y, juntando os pontos no gráfico


# In[ ]:


xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]
pontos


# In[ ]:


Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)
Z


# In[ ]:


import matplotlib.pyplot as plt
plt.contourf(xx,yy,Z, alpha = 0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c= teste_y, s=1)


# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

SEED = 5
np.random.seed(SEED)

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, 
                                                        test_size = 0.25,
                                                        stratify = y)
print("Treinaremos com %d elementos e testaremos %d elementos" % (len(treino_x), len(teste_x)))


modelo = SVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print('A acurácia é de %.2f%%' % acuracia)


# In[ ]:


x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()

pixels = 100
eixo_x = np.arange (x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange (y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)

import matplotlib.pyplot as plt
plt.contourf(xx,yy,Z, alpha = 0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c= teste_y, s=1)


# In[ ]:


#criando uma escala nova com os valores de escola do treino_x
from sklearn.preprocessing import StandardScaler 

SEED = 5
np.random.seed(SEED)

raw_treino_x, raw_teste_x, treino_y, teste_y = train_test_split(x, y, 
                                                        test_size = 0.25,
                                                        stratify = y)
print("Treinaremos com %d elementos e testaremos %d elementos" % (len(treino_x), len(teste_x)))

scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = SVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print('A acurácia é de %.2f%%' % acuracia)


# In[ ]:


data_x = teste_x[:,0]
data_y = teste_x[:,1]

x_min = data_x.min()
x_max = data_x.max()
y_min = data_y.min()
y_max = data_y.max()

pixels = 100
eixo_x = np.arange (x_min, x_max, (x_max - x_min) / pixels)
eixo_y = np.arange (y_min, y_max, (y_max - y_min) / pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

Z = modelo.predict(pontos)
Z = Z.reshape(xx.shape)

import matplotlib.pyplot as plt
plt.contourf(xx,yy,Z, alpha = 0.3)
plt.scatter(data_x, data_y, c= teste_y, s=1)


# In[ ]:




