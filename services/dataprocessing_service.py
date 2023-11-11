from datetime import datetime
from database.repositories import Repository
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import json
import os


class DataProcessingService :
  DIRECTORY = './app/documents/'
  HOST = "http://127.0.0.1:8000/"
  def __init__(self) :
    self.dataFrame = None
    self._dataBase = Repository() # type: ignore
    
  async def saveFile(self, file,extension):
    try:
      filePath,fileFullName = await self._saveFile(file)
      self._createDateFrame(extension,filePath)
      self._loadDataFrame(fileFullName)      
      return f"{fileFullName}"
    except Exception as error:
        raise error
      
  def _createDateFrame(self,extension,filePath) :
    if extension == 'xlsx' :
      self._createDataFrameExcel(filePath)
    else:
      self._createDataFrameCsv(filePath)
      
  def _loadDataFrame(self,fileName)-> str :
    try:
      if self.dataFrame is None:
        raise ValueError("Dataframe is None")
      dataToInsert = self.dataFrame.to_json(orient='records')
      return self._dataBase.insertDataframe(json.loads(dataToInsert),fileName)
    except Exception as error: 
      raise error   
      
  async def _saveFile(self, file):
    try: 
      fileContent = await file.read()
      fileName = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + file.filename
      filePath = os.path.join( self.DIRECTORY, fileName)
      with open(filePath, "wb") as f:
        f.write(fileContent)
      return filePath,fileName.split('.')[0]
    except Exception as error:
      raise error
    
  def _createDataFrameExcel(self,path):
    try:
      self._setDataFrame(pd.read_excel(path))
    except Exception as error: 
      raise error
    
  def _createDataFrameCsv(self,path):
    try:
      self._setDataFrame(pd.read_csv(path, delimiter=","))
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
  
  def processMissingData(self,dataset,type_missing_data):
    try:
      if self.dataFrame is None:
        return None
      fileFullName =f"{dataset}_clean"
      if type_missing_data == 1: self._missingDataDiscard()     
      if type_missing_data == 2: self._missingDataImputation()      
      return self._loadDataFrame(fileFullName)
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
    
  def basicDescribes(self,dataset):
    if self.searchFile(dataset):
      dict = self._basicDescribes()
      return dict
    return None
    
  