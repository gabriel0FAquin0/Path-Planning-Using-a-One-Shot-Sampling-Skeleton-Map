#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os
from matplotlib import image


# In[2]:


from torch.utils.data import DataLoader
from torch.autograd import Variable
import copy
import numpy as np
np.set_printoptions(precision=16)


# In[3]:


#!/usr/bin/env python
# coding: utf-8

# In[18]:


# from torch.utils.data import Dataset
# from torchvision import transforms
# import pandas as pd
# import os
# from matplotlib import image


# In[19]:


class maps2dDataSet_dual(Dataset):
    
    def __init__(self, csv_file, data_dir_1):
        '''run once when instantiating the dataset object
        data_dir(string): Directory with all the images
        transform(callable, optional): Optional transform to be 
        applied on a sample'''
        
        self.data_dir_1=data_dir_1
        self.landmaks=pd.read_csv(csv_file)
        self.transform_1=transforms.Compose([transforms.ToTensor(),
                                             transforms.Pad(padding=20, fill=1, padding_mode='constant')])      

    def __len__(self):
        '''return the number of samples in our dataset'''
        return len(self.landmaks) #equall to data from dir_2
    
    def __getitem__(self, idx):
        '''loads and returns  a sample from the dataset at the given index idx'''
        data_path_1=os.path.join(self.data_dir_1+'map'+str(idx)+'.png')
        
        #using matplotlib
        data_1=image.imread(data_path_1)
        data=self.transform_1(data_1)
        
        return data #retur tensor to (2,64,64)


# In[21]:


class maps2dDataSet_target(Dataset):
    
    def __init__(self, csv_file, data_dir_1):
        '''run once when instantiating the dataset object
        data_dir(string): Directory with all the images
        transform(callable, optional): Optional transform to be 
        applied on a sample'''
        
        self.data_dir_1=data_dir_1
        self.landmaks=pd.read_csv(csv_file)
        self.transform_2=transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        '''return the number of samples in our dataset'''
        return len(self.landmaks) #equall to data from dir_2
    
    def __getitem__(self, idx):
        '''loads and returns  a sample from the dataset at the given index idx'''
        data_path_1=os.path.join(self.data_dir_1+'map'+str(idx)+'.png')
        
        #using matplotlib
        data_1=image.imread(data_path_1)
        data=self.transform_2(data_1)
        return data


# In[4]:


#!/usr/bin/env python
# coding: utf-8

# In[5]:


# import torch
# import torchvision
# from torch import nn


# In[6]:


class UNet_maps(nn.Module):
    def __init__(self):
        super(UNet_maps, self).__init__()
        
        self.encoder_b1=nn.Sequential(
            
            #********BLOCK 1*****************************************
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3,
                     stride=1, padding=0, bias=True),
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3,
                     stride=1, padding=0),
            nn.ReLU(True),
        )
            #********************************************************
        
        self.encoder_b2=nn.Sequential(    
            #********BLOCK 2*****************************************
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3,
                     stride=1, padding=0),
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3,
                     stride=1, padding=0),
            nn.ReLU(True),
            #********************************************************
        )
        
        self.encoder_b3=nn.Sequential(    
            #********BLOCK 3*****************************************
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(in_channels=4, out_channels=6, kernel_size=3,
                     stride=1, padding=0),
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3,
                     stride=1, padding=0),
            nn.ReLU(True),
            #********************************************************
        )
        
            
            #********BLOCK 4****************************************
        self.decon_b4=nn.ConvTranspose2d(in_channels=6, out_channels=3,
                           kernel_size=2, stride=2, padding=0)
                         
        self.decon_b4_1=nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=4, kernel_size=3,
                      stride=1, padding=0),
            nn.ReLU(True),
        
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3,
                      stride=1, padding=0),
            nn.ReLU(True),

        )           
            #********************************************************
            
            #********BLOCK 5****************************************
        self.decon_b5=nn.ConvTranspose2d(in_channels=4, out_channels=2,
                           kernel_size=2, stride=2, padding=0)

        self.decon_b5_1=nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3,
                      stride=1, padding=0),
            nn.ReLU(True),
        
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3,
                      stride=1, padding=0),
            nn.ReLU(True),
            
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1,
                      stride=1, padding=0),
            nn.Sigmoid()

        )           
            #********************************************************
            
    def forward(self, x):
        x_b1=self.encoder_b1(x)
        
        x_b2=self.encoder_b2(x_b1)
        
        x_b3=self.encoder_b3(x_b2)
        
        x=self.decon_b4(x_b3)

        x_cc_b4=torchvision.transforms.CenterCrop((38,38))(x_b2)
        x=torch.cat((x,x_cc_b4),1) #dont work with nn
        
        x=self.decon_b4_1(x)
        
        x=self.decon_b5(x)

        x_cc_b5=torchvision.transforms.CenterCrop((68,68))(x_b1)
        x=torch.cat((x,x_cc_b5),1) 
        
        x=self.decon_b5_1(x)
        
        return x


# In[5]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
# import torch
# from torch import nn
# import model_6ii_thin as model6ii #now in main 

class init_parameters():
# class parameters(param_dict):
    
    def __init__(self):
        self.param_dict={
            'label':'0',
            're_training':False, 
            'item_block':1, 
            'item_xp':0,
            'last_epoch':0,
            'processingUnit':'CPU',
            'lossFunction':'MSELoss',
            'learningRate':1e-3,
            'epochs':5,
            'batchSize':4096,
            'weightInitialize':'uniform',
            'optimizer':'Adam',
            'seed':0
        }
    
    def init_weights(self,m):
        
        torch.manual_seed(self.param_dict.get('seed'))
        if type(m) in {nn.Conv2d, nn.ConvTranspose2d}:
            if self.param_dict.get('weightInitialize')=='uniform':
                nn.init.uniform_(m.weight,a=-10,b=10)
                nn.init.uniform_(m.bias,a=-10,b=10) if m.bias is not None else print('uninitialized weights in:',m) 
            elif self.param_dict.get('weightInitialize')=='normal':
                nn.init.normal_(m.weight,mean=0.0,std=1.0)
                nn.init.normal_(m.bias,mean=0.0,std=1.0) if m.bias is not None else print('uninitialized weights in:',m)
            elif self.param_dict.get('weightInitialize')=='constant':
                nn.init.constant_(m.weight,val=0.0)
                nn.init.constant_(m.bias,val=0.0) if m.bias is not None else print('uninitialized weights in:',m)
            else:
                print('uninitialized weights in:',m)
                   
    
    def heInitialize(self,model):
        
        torch.manual_seed(self.param_dict.get('seed'))
        for idx,m in enumerate(model.modules()):
            if idx==30: #(for sigmoid)
                fan_in, fan_out=nn.init._calculate_fan_in_and_fan_out(m.weight)
                nn.init.xavier_normal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias,val=0.0) if m.bias is not None else print('uninitialized weights in:',m)
    
            elif idx==2 or idx==4 or idx==8 or idx==10 or idx==14 or idx==16 or idx==20 or idx==22 or idx==26 or idx==28: #(for ReLU)
                fan_in, fan_out=nn.init._calculate_fan_in_and_fan_out(m.weight)
                nn.init.normal_(m.weight,mean=0.0,std=np.sqrt(2/fan_in))
                nn.init.constant_(m.bias,val=0.0) if m.bias is not None else print('uninitialized weights in:',m)
    
            elif idx==18 or idx==24: #(for convTranspose2d)
                fan_in, fan_out=nn.init._calculate_fan_in_and_fan_out(m.weight)
                nn.init.normal_(m.weight,mean=0.0,std=np.sqrt(2/fan_in))        
                nn.init.constant_(m.bias,val=0.0) if m.bias is not None else print('uninitialized weights in:',m)
    
    def WSLoss(_input, _target):
        '''own loss\n_input:input tensor\n_target:tensor\n c1:weight coefficient 0<=c1<=1'''
        c1=0.8
        err=c1*((1-_target)*torch.pow((_target-_input),2))+(1-c1)*(_target*torch.pow((_target-_input),2))
        err_m=torch.mean(err)
        return err_m
    
    def main_parameters(self):
        
        #charge model
        if self.param_dict.get('processingUnit')=='GPU':
            model=UNet_maps().cuda()# class inside class
        elif self.param_dict.get('processingUnit')=='CPU':
            model=UNet_maps()
        else:
            print('model not loaded')
            
        #initialize weights
        if self.param_dict.get('re_training')==False:
            print('Initialize weights...')
            if self.param_dict.get('weightInitialize')=='heInitialize':
                self.heInitialize(model)
            else:
                model.apply(self.init_weights)##?
        else:
            print('Load weights...')
        
        #learning rate
        learning_rate=self.param_dict.get('learningRate')
        
        #optimizer
        if self.param_dict.get('optimizer')=='Adam':
            self.param_dict['optimizer']=torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif self.param_dict.get('optimizer')=='SGD':
            self.param_dict['optimizer']=torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)
        else:
            print('unitialize optimizer')
            
        #loss function
        if self.param_dict.get('lossFunction')=='MSELoss':
            self.param_dict['lossFunction'] = nn.MSELoss()
        elif self.param_dict.get('lossFunction')=='BCELoss':
            self.param_dict['lossFunction'] = nn.BCELoss()
        elif self.param_dict.get('lossFunction') == 'WSLoss':
            self.param_dict['lossFunction']=init_parameters.WSLoss##?
        else:
            print('loss function not loaded')
        
#         print('Model:\n',type(model),'\n'+'Parameters:\n',type(self.param_dict))
        
        return model,self.param_dict


# In[6]:


class main_training:
    '''to train the model instantation and load dataset'''
    
    def __init__(self):
        
        self.exp=init_parameters()
        
        self.dataset=maps2dDataSet_dual(csv_file=
                                   'Data\\64x64_Tr10000_v2\\maps_parameters.csv',
                                   data_dir_1=
                                   'Data\\64x64_Tr10000_v2\\')
        self.dataset_target=maps2dDataSet_target(csv_file=
                                            'Data\\64x64_Tr10000_v2\\maps_parameters.csv',
                                            data_dir_1=
                                            'Data\\64x64_Tr10000_v2_thinn\\')
     
        self.dataset_test=maps2dDataSet_dual(csv_file=
                                        'Data\\64x64_Tst2500_v2\\maps_parameters_Tst_v2.csv',
                                        data_dir_1=
                                        'Data\\64x64_Tst2500_v2\\')
        self.dataset_test_target=maps2dDataSet_target(csv_file=
                                                 'Data\\64x64_Tst2500_v2\\maps_parameters_Tst_v2.csv',
                                                 data_dir_1=
                                                 'Data\\64x64_Tst2500_v2_thinn\\')
  

    def validation(self,model_test, criterion):
        
        '''for validation after each eepoch\n
            receive model'''
        model_test.eval()
        #     torch.set_grad_enabled(False)

        stackloss_numpy_test=[]

        for i in range(len(self.dataset_test)): #TRAINING DATASET
        ##########input
            input_test= self.dataset_test[i].cuda() #to tensor
            input_test=input_test.reshape(1,1,104,104)#n_samples, chanels, height, width
            input_target=self.dataset_test_target[i].cuda()
            input_target=input_target.reshape(1,1,64,64)
        ########## run model
            with torch.no_grad():#disable grad
                output_test=model_test(input_test)
        #########Error
            loss_test=criterion(input_target, output_test)
            loss_test_cpu=loss_test.to('cpu') #to numpy
            loss_test_cpu=loss_test_cpu.detach()
            loss_test_cpu=loss_test_cpu.numpy()
            stackloss_numpy_test.append(loss_test_cpu) 

        array_loss_test=np.array(stackloss_numpy_test) #to array
        validation_mean=array_loss_test.mean() #save the mean of the 2500 examples
    #     print('Validation model:',validation_mean)


        model_test.train()
    #     torch.set_grad_enabled(True)
    #     print(type(model_test))

        return validation_mean

    def set_xp(self,b):
        
        WI=b.get('weightInitialize')
        Op=b.get('optimizer')
        LR=b.get('learningRate')
        LF=b.get('lossFunction')
        
        list_epochs=[]
        list_lossTraining=[]
        list_lossValidation=[]
        list_WI=[]
        list_Op=[]
        list_LR=[]
        list_LF=[]
        count=0
        for i in range(LF.size):
            for j in range(LR.size):
                for k in range(Op.size):
                    for l in range(WI.size):
                        print('Training model: ',count)
                        b.update({'label':str(count),
                              'lossFunction':str(LF[i]) if LF.size>1 else str(LF),
                              'learningRate':float(LR[j]) if LR.size>1 else float(LR),
                              'optimizer':str(Op[k]) if Op.size>1 else str(Op),
                              'weightInitialize':str(WI[l]) if WI.size>1 else str(WI)})
                        #print('parameters:',b)
                        #***************************************************************************
                        xp_epoch, xp_lossTraining, xp_lossValidation=self.training(b) ######training
                        #***************************************************************************
                        count+=1
                        print('Finish model: ',str(count))
                        #print('best epoch:',xp_epoch,
                        #      'loss in trainig:',xp_lossTraining,
                        #      'lost in validation:',xp_lossValidation)
                        #save array results
                        list_epochs.append(xp_epoch)
                        list_lossTraining.append(xp_lossTraining)
                        list_lossValidation.append(xp_lossValidation)
                        list_WI.append(b.get('weightInitialize'))
                        list_Op.append(b.get('optimizer'))
                        list_LR.append(b.get('learningRate'))
                        list_LF.append(b.get('lossFunction'))
        #save as csv
        headers={'lossFunction':list_LF,'learningRate':list_LR,'optimizer':list_Op,'weightInitialize':list_WI,'best epoch':list_epochs,'Training':list_lossTraining,'Validation':list_lossValidation}
        
        df=pd.DataFrame(headers)
        df.to_csv('results\\Table.csv')
        print('Finish set of experiments')                
                        
    def training(self,b):
        '''Training\n b:Dictionary of parameters'''
        
        #update parameters
        for key_b, value_b in b.items():
            if key_b in self.exp.param_dict:
                self.exp.param_dict.update({key_b:value_b})
                
        model, parameters = self.exp.main_parameters()
        #print('Parameters:\n',parameters)
        #visualize some weights
        #Layers=[x.data for x in model.parameters()]
        #print(Layers[0])
        
        label=parameters.get('label')
        num_epochs=parameters.get('epochs')
        criterion = parameters.get('lossFunction')
        batch_size=parameters.get('batchSize')
        learning_rate=parameters.get('learningRate')
        optimizer=parameters.get('optimizer')
        
        if parameters.get('re_training')==True:
            print('Reasuming training...')
            item_block=parameters.get('item_block')
            item_xp=parameters.get('item_xp')
            #path_check_point='C:\\Users\\Gabri\\Documents\\MotionPlanning_MachineLearning\\Codigo\\autoEncoder\\arch6iiClass\\results\\block'+str(item_block)+'\\exp_'+str(item_xp)+'.pth'
            #path_check_point='C:\\Users\\Gabri\\Documents\\MotionPlanning_MachineLearning\\Codigo\\autoEncoder\\arch6iiClass\\results\\reasuming_training\\1000-4000ep\\block'+str(item_block)+'xp0'+'\\exp_'+str(item_xp)+'.pth'
            path_check_point='C:\\Users\\Gabri\\Documents\\MotionPlanning_MachineLearning\\Codigo\\autoEncoder\\arch6iiClass\\results\\reasuming_training\\4000-7000ep\\block'+str(item_block)+'xp0'+'\\exp_'+str(item_xp)+'.pth'
            print(path_check_point)
            checkpoint=torch.load(path_check_point)
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch=parameters.get('last_epoch')

        #print(type(num_epochs),type(criterion),type(batch_size),type(learning_rate),type(optimizer))
            
        dataloader=DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        dataloader_target=DataLoader(self.dataset_target, batch_size, shuffle=False)   
        
        # stack_loss=[]
        benchmark_loss=100 #what is a good value 

        for epoch in range(num_epochs):
            for data, data_target in zip(dataloader,dataloader_target):

                #img, _ = data
                img = data
                img = Variable(img).cuda()

                img_target = data_target
                img_target = Variable(img_target).cuda()
                #=============forward==================
                output = model(img)
                #print('output',type(output),output.dim(),output.size(),output.device)

                loss = criterion(output, img_target)
                #print('loss',loss, type(loss),loss.dim(),loss.size(), loss.device)

                #=============backward=================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            #==============validation====================
            #call function validation
            validation_loss=self.validation(model, criterion)
            #================hold========================
#             stack_loss.append(loss.cpu())# append as cpu tensor

            if validation_loss<benchmark_loss:
                #save the model
                best_model=copy.deepcopy(model) #copy in gpu
                best_model.to('cpu') #copy in cpu
                #save the epoch
                best_epoch=epoch
                #save optimizer
                best_optimizer=optimizer.state_dict()
                #save loss
                benchmark_loss=validation_loss #tensor cuda
                 
            #==============log==========================
            if epoch % 100 == 0:
                loss_np=loss
                loss_np=loss_np.to('cpu')
                loss_np=loss_np.detach()
                loss_np=loss_np.numpy()
                
                print("epochs:",epoch,'        loss_training:',loss_np,'        loss_validation:',validation_loss)   
        #=================save======================
         #     if epoch % 1 == 0:
        torch.save({'epoch':best_epoch+last_epoch,'state_dict':best_model.state_dict(),'optimizer_state_dict':best_optimizer,'loss':benchmark_loss},
        'results\\exp_'+label+'.pth')
#         print('Final:\n')        
#         print("best_epoch:",best_epoch,' loss:',benchmark_loss)
        
        return best_epoch+last_epoch, loss_np, validation_loss

    