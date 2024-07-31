import copy
import os
import pickle as pkl
from datetime import datetime
from pathlib import Path

import pymongo
from bson.objectid import ObjectId
import gridfs

from models.LSTMRegressor import LSTMRegressor
from models.StackedLSTMRegressor import StackedLSTMRegressor
from models.nbeats import NBeatsNet
from models.nbeatsx.nbeats_model import NBeatsX
from util import Config


def _from_gridfs(database_name, object_id):
    try:
        client = pymongo.MongoClient(Config.mongodb_url,
                                     username=Config.mongo_db_user,
                                     password=Config.mongo_db_password)
        db = client[database_name]
        fs = gridfs.GridFS(db)

        if type(object_id) == str:
            object_id = ObjectId(object_id)

        retrieved = fs.get(object_id).read()
        return pkl.loads(retrieved)

    except Exception as e:
        print(f'Error while retrieving from GridFS mongoDB: {e}')


def _write_to_db(to_insert):
    client = None
    try:
        client = pymongo.MongoClient(Config.mongodb_url,
                                     username=Config.mongo_db_user,
                                     password=Config.mongo_db_password)
        db = client[Config.mongodb_db_name]
        collection = db[Config.mongodb_collection_name]

        fs = gridfs.GridFS(db)
        id = fs.put(to_insert['model']['model_state_dict'])
        to_insert['model']['model_state_dict'] = id

        id = fs.put(to_insert['model']['pickled_scalers'])
        to_insert['model']['pickled_scalers'] = id

        collection.insert_one(to_insert)
    except Exception as e:
        print(f'Error while inserting into mongoDB: {e} \nSaving document to disk..')

        to_insert.pop('_id', None)  # remove id from document
        os.makedirs('./failed-inserts', exist_ok=True)
        date_string = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        file_path = Path(f'./failed-inserts/{date_string}.pkl')
        c = 0

        while file_path.is_file():
            file_path = Path(f'./failed-inserts/{date_string}_({c}).pkl')
            c += 1

        with open(file_path, 'wb') as f:
            pkl.dump(to_insert, f)

        print(f'\tsaved to {file_path.name}')
    finally:
        if client:
            client.close()


def _unpickle_saved_dict(file_path):
    with open(file_path, 'rb') as handle:
        loaded_dict = pkl.load(handle)
    return loaded_dict


def _remove_collection(collection_name, mongodb_name, sanity_check=True):
    print(f'deleting collection "{collection_name}" from database "{mongodb_name}".')

    try:
        client = pymongo.MongoClient(Config.mongodb_url,
                                     username=Config.mongo_db_user,
                                     password=Config.mongo_db_password)
        db = client[mongodb_name]
        collection = db[collection_name]
        fs = gridfs.GridFS(db)

        cursor = collection.find({})
        document_counter = 0

        for document in cursor:
            document_counter = document_counter + 1
            model_id = document['model']['model_state_dict']
            scaler_id = document['model']['pickled_scalers']

            if not sanity_check:
                fs.delete(model_id)
                fs.delete(scaler_id)

        print(f'{document_counter} found')
        if not sanity_check:
            collection.drop()

    except Exception as e:
        print(f'Error while deleting from mongoDB: {e}')


def get_collection_object(database_name=None, collection_name=None):
    client = pymongo.MongoClient(Config.mongodb_url,
                                 username=Config.mongo_db_user,
                                 password=Config.mongo_db_password)

    if database_name is None:
        database_name = Config.mongodb_db_name
    db = client[database_name]

    if collection_name is None:
        collection_name = Config.mongodb_collection_name
    collection = db[collection_name]

    return collection

def save_model_data(experiment_config, model_to_save, scalers, metrics, evaluation_samples):
    copied_experiment_config = copy.deepcopy(experiment_config)
    copied_experiment_config.pop('scalers', None)
    copied_experiment_config.pop('train_dataset', None)
    copied_experiment_config.pop('val_dataset', None)
    copied_experiment_config.pop('test_dataset', None)
    copied_experiment_config.pop('repository_version', None)

    copied_model_hparams = copy.deepcopy(model_to_save.hparams)
    copied_model_hparams.pop('scalers', None)

    to_insert = {
        'experiment_config': copied_experiment_config,
        'repository_version': experiment_config['repository_version'],
        'model':
            {
                'model_state_dict': pkl.dumps(model_to_save.state_dict()),
                'model_hparams': copied_model_hparams,
                'pickled_scalers': pkl.dumps(scalers),
            },
        'metrics': metrics,
        'evaluation_samples': evaluation_samples
    }

    _write_to_db(to_insert)


def init_LSTMRegressor(doc, model_state_dict, scalers):
    doc['model']['model_state_dict'] = model_state_dict
    doc['experiment_config']['scalers'] = scalers

    model_kwargs = doc['model']['model_hparams']
    model_kwargs.pop('scalers', None)
    loaded_model = LSTMRegressor(scalers=scalers, **model_kwargs)

    loaded_model.load_state_dict(model_state_dict)

    return loaded_model


def init_stackedLSTMRegressor(doc, model_state_dict, scalers):
    doc['model']['model_state_dict'] = model_state_dict
    doc['experiment_config']['scalers'] = scalers

    model_kwargs = doc['model']['model_hparams']
    model_kwargs.pop('scalers', None)
    loaded_model = StackedLSTMRegressor(scalers=scalers, **model_kwargs)
    loaded_model.load_state_dict(model_state_dict)

    return loaded_model


def init_NBEATSx(doc, model_state_dict, scalers):
    doc['model']['model_state_dict'] = model_state_dict
    doc['experiment_config']['scalers'] = scalers

    model_kwargs = doc['model']['model_hparams']
    model_kwargs.pop('scalers', None)
    loaded_model = NBeatsX(scalers=scalers, **model_kwargs)

    loaded_model.load_state_dict(model_state_dict)
    return loaded_model


def init_NBEATS(doc, model_state_dict, scalers, device='cuda'):
    doc['model']['model_state_dict'] = model_state_dict
    doc['experiment_config']['scalers'] = scalers

    model_kwargs = doc['model']['model_hparams']
    model_kwargs.pop('device', None)
    model_kwargs.pop('scalers', None)
    loaded_model = NBeatsNet(scalers=scalers, device=device, **model_kwargs)

    loaded_model.load_state_dict(model_state_dict)
    return loaded_model


def remove_new_metrics(database_name, collection_name):
    collection = get_collection_object(database_name=database_name, collection_name=collection_name)
    collection.update_many({}, {'$unset': {'metrics_new': 1}})


def update_document(database_name, collection_name, doc_id, new_data):
    """
    Updates a given entry in the MongoDB with the given new data. new_data has to be a dictionary!

    :param database_name:
    :param collection_name:
    :param doc_id:
    :param new_data:
    :return:
    """
    collection = get_collection_object(database_name=database_name, collection_name=collection_name)
    collection.update_one({"_id": doc_id}, {"$set": new_data})


def load_model(database_name, model_doc):
    model_type = model_doc['experiment_config']['model']

    gridfs_id_state_dict = model_doc['model']['model_state_dict']
    gridfs_id_scalers = model_doc['model']['pickled_scalers']

    model_state_dict = _from_gridfs(database_name, gridfs_id_state_dict)
    scalers = _from_gridfs(database_name, gridfs_id_scalers)

    if model_type == 'LSTMRegressor':
        return init_LSTMRegressor(model_doc, model_state_dict, scalers), scalers
    elif model_type == 'StackedLSTMRegressor':
        return init_stackedLSTMRegressor(model_doc, model_state_dict, scalers), scalers
    elif model_type == 'NBEATS':
        return init_NBEATS(model_doc, model_state_dict, scalers), scalers
    elif model_type == 'NBEATS-trend-seasonality':
        return init_NBEATS(model_doc, model_state_dict, scalers), scalers
    elif model_type == 'NBEATSx':
        return init_NBEATSx(model_doc, model_state_dict, scalers), scalers
    else:
        raise Exception("Not implemented yet!")


def load_collection(database_name, collection_name):
    collection = get_collection_object(database_name=database_name, collection_name=collection_name)
    return list(collection.find({}))


def load_first(database_name, collection_name, sort_parameters=None):
    if sort_parameters is None:
        sort_parameters = {}

    client = pymongo.MongoClient(Config.mongodb_url,
                                 username=Config.mongo_db_user,
                                 password=Config.mongo_db_password)
    db = client[database_name]
    collection = db[collection_name]
    return collection.find_one({}, sort=sort_parameters)


def load_doc(day=None, idx=None, model=None):
    collection = get_collection_object()

    query = {}

    if day:
        query['experiment_config.starting_date'] = day
    if idx:
        query['experiment_config.experiment_idx'] = idx
    if model:
        query['experiment_config.model'] = model

    doc = collection.find_one(query)
    return doc


def get_document_from_objectID(object_id, database_name=None, collection_name=None):
    collection = get_collection_object(database_name, collection_name)
    query = {"_id": ObjectId(object_id)}
    return collection.find_one(query)


def load_and_save_experiment_document(file_path, day=None, idx=None, model=None):
    doc = load_doc(day=day, idx=idx, model=model)

    # save to pickled experiment document containing scalers and weights for model
    with open(file_path, 'wb') as handle:
        pkl.dump(doc, handle, protocol=pkl.HIGHEST_PROTOCOL)
