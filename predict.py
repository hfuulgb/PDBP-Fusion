# _*_ coding:utf-8 _*_
from encode_schema import read_seq_onehot,get_seq_concolutional_array
from keras.models import load_model
import numpy as np
from datetime import datetime

class Pred():
    def __init__(self,model_path='./model_save/model_'):
        self.item_model_path = model_path + str(0) + '.hdf5'
        #self.model = load_model(item_model_path)

    def sample_predict(self,string,str_len=850):
        # make sure the input sequnce less than 850, otherwise drop string when length excell 850
        if len(string)>=str_len:
            string=string[:str_len]
        else:
            string=string+'Z'*(str_len-len(string))
        #print('len: ',str_len)
        assert len(string)==str_len

        #向量化
        #print(get_seq_concolutional_array(string)[:5,:])
        string_vector=get_seq_concolutional_array(string)
        string_vector=np.expand_dims(string_vector,0)

        #load model and predict
        result=[]
        self.model = load_model(self.item_model_path)
        predict=self.model.predict(string_vector)[0]
        print('prediction is {}'.format(predict))
        result.append(predict)
        print(result)

        return result

if __name__=='__main__':
    Pr=Pred()
    start = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    Pr.sample_predict(string='GSHMSDKPLTKTDYLMRLRRCQTIDTLERVIEKNKYELSDNELAVFYSAADHRLAELTMNKLYDKIPSSVWKFIRZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZABC')
    end = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    print('start: %s' % start)
    print('end: %s' % end)
    pass
