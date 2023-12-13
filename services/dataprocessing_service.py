from datetime import datetime
from database.repositories import Repository
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import json
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
import io



class DataProcessingService :
  DIRECTORY = './app/documents/'
  HOST = "http://127.0.0.1:8000/"
  def __init__(self) :
    self.dataFrame = None
    self._dataBase = Repository("main")
    
  async def saveFile(self, file,extension):  
    try:
      fileContent = file.read()
      fileName = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + file.filename
      
      data = pd.read_csv(file.file)
      data = data.fillna("NaN")

      dataToInsert = data.to_json(orient='records')
      insertadoId = str(self._dataBase.insertDataframe(json.loads(dataToInsert),fileName))


      return insertadoId
    
    except Exception as error:
        raise error
    
  def saveUpdateDF(self, data, filename):
    try:
      insertadoId = str(self._dataBase.insertDataframe(data, filename))
      
      return insertadoId

    except Exception as error:
      raise error
  
  def _loadDataFrame(self,fileName)-> str :
    try:
      if self.dataFrame is None:
        raise ValueError("Dataframe is None")
      dataToInsert = self.dataFrame.to_json(orient='records')
      return self._dataBase.insertDataframe(json.loads(dataToInsert),fileName)
    except Exception as error: 
      raise error   
    
  def _setDataFrame(self, dataFrame):
    self.dataFrame = dataFrame
    
  def searchFile(self,dataset):
     pathFile,extension = self._searchFile(dataset)
     if pathFile is None:
       return False
     self._createDateFrame(extension,pathFile) 
     return True

  def _describeDataset(self):
    try:
      return None if self.dataFrame is None else  self._describeData()
    except Exception as error:
      raise error

  def describeDataset(self,dataset):
    if self.searchFile(dataset):
      dict = self._describeDataset()
      return dict
    return None
 
  def _searchFile(self, dataset):
    for root, _, files in os.walk(self.DIRECTORY):      
      if f"{dataset}.csv" in files:
        return f"{os.path.join(root, dataset)}.csv" ,'csv'
      if f"{dataset}.xlsx" in files:
        return f"{os.path.join(root, dataset)}.xlsx" ,'xlsx'
    return None, any

  def _describeData(self):
    try:
      if self.dataFrame is None:
        return None
      firstRecordDict = self.dataFrame.iloc[0].to_dict()
      return {key: type(value).__name__ for key, value in firstRecordDict.items()}
        
    except Exception as error:
      raise error        
          
  def getDataset(self):
    fileNames = []
    for fileName in os.listdir(self.DIRECTORY):
      if os.path.isfile(os.path.join(self.DIRECTORY, fileName)):
          fileNames.append(fileName.split('.')[0])
    return fileNames
  
  def processMissingData(self,dataset_id, type_missing_data):
    try:
      data = self.loadDataFromMongoOnline(dataset_id)

      if data[0]:
        if type_missing_data == 1 or type_missing_data == 2:
          if type_missing_data == 1:
            newData = self._eraseRowIfExitstNull(data[1])

          if type_missing_data == 2:
            newData = self._replaceRowValueIfExitstsNull(data[1])

          filename = f"update-{self.fileNameFromMongoOnline(dataset_id)}"
          id = self.saveUpdateDF(newData, filename)
          return id

        return f"NOT VALID OPTION {type_missing_data}"

      return None

    except Exception as error:
      raise error
    
  def _missingDataDiscard(self):
    try:
        if self.dataFrame is None:
          return None
        dataFrameNN = self.dataFrame.dropna()  
        self._setDataFrame(dataFrameNN)
                 
    except Exception as error:
      print(error)
      raise error
    
  def _missingDataImputation(self):
    try:
        if self.dataFrame is None:
          return None
        numColumns = self.dataFrame.select_dtypes(np.number).columns
        objColumns = self.dataFrame.select_dtypes(object).columns
        dataFrame = self._averageImputation(self.dataFrame, numColumns)
        dataFrame = self._modeImputation(dataFrame, objColumns)
        self._setDataFrame(dataFrame)
    except Exception as error:
        raise error
        
  def _averageImputation(self, dataFrame, numColumns):
    for column in numColumns:
      media = dataFrame[column].mean()
      dataFrame[column].fillna(media, inplace=True)
    return dataFrame

  def _modeImputation(self, dataFrame, objColumns):
    for column in objColumns:
      mode = dataFrame[column].mode()[0]
      dataFrame[column].fillna(mode, inplace=True)
    return dataFrame
  
  def graphicalAnalysis(self,dataset):
    if not self.searchDatase(dataset) :
      return None, None, None
    arr =  dataset.split('_')
    if arr[len(arr)-1] != "limpio":
      raise ValueError("It is recommended to first do the treatment of missing data for the dataset '{dataset}'")
    histograms = self._histograms(dataset)
    boxplots = self._boxPlots(dataset)     
    probilityDistribution = self._probabilityDistribution(dataset) 
    return histograms, probilityDistribution, boxplots

  def _histograms(self,dataset):
    try:
      if self.dataFrame is None:
        return None
      dfNumeric = self.dataFrame.select_dtypes(np.number)
      plt.rcParams['figure.figsize'] = (19, 9)
      plt.style.use('ggplot')
      dfNumeric.hist()
      folderPath = self.createfolder("histograms")
      filePath = os.path.join(folderPath, dataset)
      plt.savefig(filePath)
      return f"{self.HOST}histograms/{dataset}.png"      
            
    except Exception as error:
      print(f"error{error} hi")
      raise error

  def _boxPlots(self,dataset):
    try:
      if self.dataFrame is None:
        return None
      dfNumeric = self.dataFrame.select_dtypes(np.number)
      plt.rcParams['figure.figsize'] = (19, 9)
      plt.style.use('ggplot')
      dfNumeric.boxplot()
      folderPath = self.createfolder("boxPlots")
      filePath = os.path.join(folderPath, dataset)
      plt.savefig(filePath)
      return f"{self.HOST}boxPlots/{dataset}.png"
    except Exception as error:
      print(f"error{error} bo")
      raise error
    
  def _probabilityDistribution(self,dataset):
    try:
      if self.dataFrame is None:
        return None
      dfNumeric = self.dataFrame.select_dtypes(np.number)
      plt.rcParams['figure.figsize'] = (19, 9)
      plt.style.use('ggplot')
      dfNumeric.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
      folderPath = self.createfolder("probabilityDistribution")
      filePath = os.path.join(folderPath, dataset)
      plt.savefig(filePath)
      return f"{self.HOST}probabilityDistribution/{dataset}.png"
    except Exception as error:
      print(f"error{error} pr")
      raise error

  def _correlationMatrix(self,dataset):
    try:
        if self.dataFrame is None:
          return None
        dfNumeric =  self.dataFrame.select_dtypes(np.number)  
        correlationMatrix = dfNumeric.astype(float).corr()
        colorMap = plt.cm.get_cmap('coolwarm')
        plt.figure(figsize=(12,12))
        plt.title('Correlation of Features', y=1.05, size=15)
        sb.heatmap(correlationMatrix,linewidths=1,vmax=1.0, square=True, cmap=colorMap,linecolor='white', annot=True)
        folderPath = self.createfolder("correlationMatrix")
        filePath = os.path.join(folderPath, dataset)
        plt.savefig(filePath)
        return f"{self.HOST}correlationMatrix/{dataset}.png"
    except Exception as error:
      print(f"error{error} ma")
      raise error
        
  def createfolder(self, folderName):
    folderPath = os.path.join(self.DIRECTORY, folderName)
    if not os.path.exists(folderPath):
      os.makedirs(folderPath)
    return folderPath
  
  def searchDatase(self,dataset):
    data =  self._dataBase.getColeccion(dataset)
    if data is None:
      return  False
    resultados = list(data)
    df = pd.DataFrame(resultados)
    self._setDataFrame(df)
    return True
    
  def _basicDescribes(self):
    try:
      if self.dataFrame is None:
        return None
      return self.dataFrame.describe().to_dict()
    except Exception as error:
      raise error    
    
  def basicDescribes(self,dataset_id):
       data = self.loadDataFromMongoOnline(dataset_id)

       if data[0]:
         pandas_df = self.reconstructData(data[1])
         export_df = self.describePandasDataFrame(pandas_df)
         export_df = self.formatPrettyResponse(export_df)
         return export_df

       return None
        

  def loadDataFromMongoOnline(self,dataset_id):
    data = self._dataBase.getCollectionByID(dataset_id)

    if data != None:
      return (True, data)
    
    return (False, None)
  
  def fileNameFromMongoOnline(self, dataset_id):
    data = self.loadDataFromMongoOnline(dataset_id)

    if data[0]:
      filename = list(data[1])
      filename = filename[0]
      filename = filename['file_name']
      return filename
    
    return None


  def reconstructData(self, data):
    objConvert = list(data)
    objConvert = objConvert[0]["data"]
    objConvert = str(objConvert)
    objConvert = objConvert.replace("\'", "\"")
    objConvert = json.loads(objConvert)

    return pd.DataFrame(objConvert)
  

  def describePandasDataFrame(self, pandas_data):
    return pandas_data.describe()
  

  def formatPrettyResponse(self, data):
      final_cols = ['Age ', 'Gender', 'BMI', 'Nausea/Vomting']

      for i in data.columns:
          if i not in final_cols:
              data = data.drop([i], axis=1)

      return data
  

  def getColumsNames(self, idataset_id):
    """
    Enter a ID to search in MongoDB and return all Coluns Names
    """
    data = self.loadDataFromMongoOnline(idataset_id)

    if data[0]:
      return self._getColumsNamesXDataType(data[1])


    return None
  

  def _getColumsNamesXDataType(self, data):
    df = self.reconstructData(data)
    _dtTypes = {}
    for i in df.columns.tolist():
      _dtTypes[i] = df[i].dtypes

    return str(_dtTypes)
  

  def _eraseRowIfExitstNull(self, data):
    data = self.reconstructData(data)
    _erase = data.eq("NaN").any(axis=1)
    data = data[~_erase]
    data = data.to_json(orient='records')
  
    return data


  def pca(self, idataset_id ):
    data= self.loadDataFromMongoOnline(idataset_id)
    if data [0]:
      return self._pca(data[1])

    return None

  def _pca(self, data):
    try:
      dataFilename = data.clone()
      filename = list(dataFilename)
      filename = filename[0]["file_name"]
      dataframes = self.reconstructData(data)
      dataframes = dataframes.dropna()
      dataframes = dataframes.select_dtypes(np.number)
      dataframes = dataframes.dropna()
      
      n_components = 2
      pca = PCA(n_components=2)
      data_pca = pca.fit_transform(dataframes)
      
      filename = "pca".join(filename).capitalize()
      pca_columns = [f'Componente_Principal_{i+1}' for i in range(n_components)]
      data_pca2 = pd.DataFrame(data_pca, columns=pca_columns)
      data_pca2 = data_pca2.to_json(orient='records')

      id = self.saveUpdateDF(data_pca2, filename)
      
     
      plt.scatter(data_pca[:,0], data_pca[:,1])
      plt.xlabel('Componente Principal 1')
      plt.ylabel('Componente Principal 2')
      
      img_buffer = io.BytesIO()
      plt.savefig(img_buffer, format='png')
      img_buffer.seek(0)
      plt.close()
      return img_buffer, id
    except Exception as error:
      raise error

    
    
    
  
  

  def _replaceRowValueIfExitstsNull(self, data):
    data = self.reconstructData(data)
    for i in data.columns:
      _data = data[i]
      if "NaN" in _data.values:
        if self.allValuesInColBeNumber(data, i):
          _mean = self.getValueOfMean(data, i)
          data[i] = data[i].replace('NaN', _mean)

    data = data.to_json(orient='records')
    return data

  def allValuesInColBeNumber(self, data, key):
    """
    retrun bool Say if all values in col be (integer or float)
    Iterate a pandas col
    if all values in col be integer or float always expect condition
    and hav inusual value is a error
    """
    _errors = 0
    for i in data[key]:
      if str(i) == "NaN":
        continue
      else:
        if type(i) is float or type(i) is int:
          continue
        _errors = _errors + 1
      
    return _errors == 0
  

  def getValueOfMean(self, data, key):
    """
    Enter a pandas dataframe 
    and calculate mean 
    """
    _errors = 0
    _lenData = 0
    _sum = 0

    for i in data[key]:
      if str(i) == "NaN":
        continue
      else:
        if type(i) is float or type(i) is int:
          _sum = _sum + float(i)
          _lenData = _lenData + 1
        else:
          _errors = _errors + 1

    if _errors == 0:
      return _sum / _lenData
    
    return 0
