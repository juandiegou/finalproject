"""
    This is a Router for a company services
"""
from bson import ObjectId
from fastapi import APIRouter, UploadFile, File, status, Response
from fastapi.responses import JSONResponse
from services import DataProcessingService
router = APIRouter()
data_processing_service = DataProcessingService()

ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

@router.post('/load')
async def load_file(file :UploadFile) -> JSONResponse:
    """_summary_

    Args:
        file (File): The file to load

    Returns:
        JSONResponse: The response of load
    """
    try:
        print("file:",file.file)
        if not file.filename:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                                content={"detail": "No file name provided"})
        extension = file.filename.split('.')[-1] 
        if extension not in ALLOWED_EXTENSIONS:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                                content={"detail": "Extension file not allowed"})
        dataset= await data_processing_service.saveFile(file,extension)
        if dataset is not None:
            print("log1:")
            return JSONResponse(status_code=status.HTTP_200_OK,
                                content={"detail": "File loaded successfully",
                                         "dataset_id": dataset})
        print("log2:")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={"detail": "Upload file error"})
    
    except Exception as error:  
        print("error:",error)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            content={"detail": "Upload file error"})
        
@router.get('/basics_statistics/{dataset_id}')
async def get_basic_statistics(dataset_id: str) -> JSONResponse:
    """_summary_

    Args:
        dataset_id (str): The id of dataset

    Returns:
        JSONResponse: The basic statistics of dataset
    """
    obj_id = ObjectId(dataset_id)
    basic_statistics = str(data_processing_service.basicDescribes(obj_id))
    # descargar en excel

    if basic_statistics is not None:
        return JSONResponse(status_code=status.HTTP_200_OK,
                            content={"detail": "Dataset retrieved successfully",
                                         "dataset": basic_statistics})
    else:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND,content={"detail": "Dataset not found"})
    
    
@router.get('/columns-describe/{dataset_id}')
async def get_columns_describe(dataset_id: str) -> JSONResponse:
    """_summary_

    Args:
        dataset_id (str): The id of dataset

    Returns:
        JSONResponse: The columns describe of dataset
    """
    obj_id = ObjectId(dataset_id)
    description = str(data_processing_service.getColumsNames(obj_id))

    if description is None:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND,
                            content={"detail": "Dataset not found"})
    return JSONResponse(status_code=200,
                        content={"detail": description})
       
@router.post('/imputation/{dataset_id}/type/{number_type}')
async def imputation_and_type(dataset_id:str, number_type: int) -> JSONResponse:
    """_summary_

    Args:
        dataset_id (int): The id of dataset
        number_type (int): The number type to imputation

    Returns:
        JSONResponse: The imputation of dataset
    """
    obj_id = ObjectId(dataset_id)
    clean_dataset = data_processing_service.processMissingData(obj_id, type_missing_data=number_type)
    
    if clean_dataset is None:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND,
                            content={"detail": "Dataset not found"})
    else:
        return JSONResponse(status_code=200,
                            content={"detail": "Dataset imputated successfully", 
                                     "dataset_id": str(clean_dataset)})
    
    
@router.post('general-univariable-graphs/{dataset_id}')
async def set_general_univariable_graphs(dataset_id: int) -> JSONResponse:
    """_summary_

    Args:
        dataset_id (int): The id of dataset

    Returns:
        JSONResponse: The general univariable graphs of dataset
    """
    histograms, probilityDistribution, boxplots = data_processing_service.graphicalAnalysis(dataset=dataset_id)
    content_response = {}
    if histograms is not None:
        content_response["histograms"] = histograms
    if probilityDistribution is not None:
        content_response["probilityDistribution"] = probilityDistribution
    if boxplots is not None:
        content_response["boxplots"] = boxplots
    return JSONResponse(status_code=200,
                        content={"detail": "grafical analisys created successfully", "data": content_response })
    
    
@router.post('univariable-graphs-class/{dataset_id}')
async def set_general_univariable_class(dataset_id: int) -> JSONResponse:
    """_summary_

    Args:
        dataset_id (int): The id of dataset

    Returns:
        JSONResponse: The general univariable class of dataset
    """
    return JSONResponse(status_code=200,
                        content={"detail": "general_univariable_class"})
    
@router.get('bivariable-graphs-class/{dataset_id}')
async def bivariable_graphs_class(dataset_id: int) -> JSONResponse:
    """_summary_

    Args:
        dataset_id (int): The id of dataset

    Returns:
        JSONResponse: The bivariable graphs class of dataset
    """
    return JSONResponse(status_code=200,
                        content={"detail": "bivariable_graphs_class"})
    
    
@router.get('multivaribale-graphs-class/{dataset_id}')
async def multivaribale_graphs_class(dataset_id: int) -> JSONResponse:
    """_summary_

    Args:
        dataset_id (int): The id of dataset

    Returns:
        JSONResponse: The multivaribale graphs class of dataset
    """
    return JSONResponse(status_code=200,
                        content={"detail": "multivaribale_graphs_class"})
    
@router.post('pca/{dataset_id}')
async def pca(dataset_id: int) -> JSONResponse:
    """_summary_

    Args:
        dataset_id (int): The id of dataset

    Returns:
        JSONResponse: The pca of dataset
    """
    return JSONResponse(status_code=200,
                        content={"detail": "pca"})
    
@router.post('train/{dataset_id}')
async def train(dataset_id: int, algorithms:str, option_train:str,normalization:str) -> JSONResponse:
    """_summary_

    Args:
        dataset_id (int): The id of dataset
        algorithms (str): The algorithms to train
        option_train (str): The option train
        normalization (str): The normalization

    Returns:
        JSONResponse: The train of dataset
    """
    return JSONResponse(status_code=200,
                        content={"detail": "train"})
    
@router.get('results/{train_id}')
async def results(train_id: int) -> JSONResponse:
    """_summary_

    Args:
        train_id (int): The id of train

    Returns:
        JSONResponse: The results of train
    """
    return JSONResponse(status_code=200,
                        content={"detail": "results"})
    
@router.get('prediction/{train_id}')   
async def prediction(train_id: int) -> JSONResponse:
    """_summary_

    Args:
        train_id (int): The id of train

    Returns:
        JSONResponse: The prediction of train
    """
    return JSONResponse(status_code=200,
                        content={"detail": "prediction"})
