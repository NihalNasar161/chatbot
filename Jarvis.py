#TechVidvan ChatBot project
import nltk, random, json , pickle
nltk.download('punkt');nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import flatten
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer=WordNetLemmatizer()
class Training:
    def __init__(self):
        #read and load the intent file
        data_file=open('C:/Users/user/Downloads/chatbot-code/chatbot-code/intents.json').read()
        self.intents=json.loads(data_file)['intents']
        self.ignore_words=list("!@#$%^&*?")
        self.process_data()

    def process_data(self):
        #fetch patterns and tokenize them into words
        self.pattern=list(map(lambda x:x["patterns"],self.intents))
        self.words=list(map(word_tokenize,flatten(self.pattern)))
        #fetch classes i.e. tags and store in documents along with tokenized patterns 
        self.classes= flatten( [[x["tag"]]*len(y) for x,y in zip(self.intents,self.pattern)])
        self.documents=list(map(lambda x,y:(x,y),self.words,self.classes))
        #lower case and filter all the symbols from words
        self.words=list(map(str.lower,flatten(self.words)))
        self.words=list(filter(lambda x:x not in self.ignore_words,self.words))
        
        #lemmatize the words and sort the class and word lists                    
        self.words=list(map(lemmatizer.lemmatize,self.words))
        self.words=sorted(list(set(self.words)))
        self.classes=sorted(list(set(self.classes)))

    def train_data(self):
        #initialize and set analyzer=word as we want to vectorize words not characters
        cv=CountVectorizer(tokenizer=lambda txt: txt.split(),analyzer="word",stop_words=None)
        #create the training sets for model
        training=[]
        for doc in self.documents:
            #lower case and lemmatize the pattern words
            pattern_words=list(map(str.lower,doc[0]))
            pattern_words=' '.join(list(map(lemmatizer.lemmatize,pattern_words)))

            #train or fit the vectorizer with all words
            #and transform into one-hot encoded vector
            vectorize=cv.fit([' '.join(self.words)])
            word_vector=vectorize.transform([pattern_words]).toarray().tolist()[0]

            #create output for the respective input
            #output size will be equal to total numbers of classes
            output_row=[0]*len(self.classes)

            #if the pattern is from current class put 1 in list else 0
            output_row[self.classes.index(doc[1])]=1
            cvop=cv.fit([' '.join(self.classes)])
            out_p=cvop.transform([doc[1]]).toarray().tolist()[0]

            #store vectorized word list long with its class
            training.append([word_vector,output_row])

        #shuffle training sets to avoid model to train on same data again
        random.shuffle(training)
        training=np.array(training,dtype=object)
        train_x=list(training[:,0])#patterns
        train_y=list(training[:,1])#classes
        print(train_y)
        return train_x,train_y 

    def build(self):
        #load the data from train_data function
        train_x,train_y = self.train_data()
        
        ##Create a Sequential model with 3 layers. 
        model=Sequential()
        #input layer with latent dimension of 128 neurons and ReLU activation function 
        model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
        model.add(Dropout(0.5)) #Dropout to avoid overfitting
        #second layer with the latent dimension of 64 neurons
        model.add(Dense(64,activation='relu')) 
        model.add(Dropout(0.5))
        #fully connected output layer with softmax activation function
        model.add(Dense(len(train_y[0]),activation='softmax')) 
        '''Compile model with Stochastic Gradient Descent with learning rate  and
           nesterov accelerated gradient descent'''
        sgd=SGD(lr=1e-2,momentum=0.9,nesterov=True)
        model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
        #fit the model with training input and output sets
        hist=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=10,verbose=1)
        #save model and words,classes which can be used for prediction.
        model.save('chatbot_model.h5',hist)
        pickle.dump({'words':self.words,'classes':self.classes,'train_x':train_x,'train_y':train_y},
                    open("training_data","wb"))

#train the model
Training().build()

import nltk, random, json , pickle
#nltk.download('punkt');nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import CountVectorizer

lemmatizer=WordNetLemmatizer()
context={};
class Testing:
    def __init__(self):
        #load the intent file
        self.intents = json.loads(open('C:/Users/user/Downloads/chatbot-code/chatbot-code/intents.json').read())
        #load the training_data file which contains training data
        data=pickle.load(open("training_data","rb"))
        self.words=data['words']
        self.classes=data['classes']
        self.model=load_model('chatbot_model.h5')
        #set the error threshold value
        self.ERROR_THRESHOLD=0.5
        self.ignore_words=list("!@#$%^&*?")
        
    def clean_up_sentence(self,sentence):
        #tokenize each sentence (user's query)
        sentence_words=word_tokenize(sentence.lower())
        #lemmatize the word to root word and filter symbols words
        sentence_words=list(map(lemmatizer.lemmatize,sentence_words))
        sentence_words=list(filter(lambda x:x not in self.ignore_words,sentence_words))
        return set(sentence_words)

    def wordvector(self,sentence):
        #initialize CountVectorizer
        #txt.split helps to tokenize single character
        cv=CountVectorizer(tokenizer=lambda txt: txt.split())
        sentence_words=' '.join(self.clean_up_sentence(sentence))
        words=' '.join(self.words)

        #fit the words into cv and transform into one-hot encoded vector
        vectorize=cv.fit([words])
        word_vector=vectorize.transform([sentence_words]).toarray().tolist()[0]
        return(np.array(word_vector)) 

    def classify(self,sentence):
        #predict to which class(tag) user's query belongs to
        results=self.model.predict(np.array([self.wordvector(sentence)]))[0]
        #store the class name and probability of that class 
        results = list(map(lambda x: [x[0],x[1]], enumerate(results)))
        #accept those class probability which are greater then threshold value,0.5
        results = list(filter(lambda x: x[1]>self.ERROR_THRESHOLD ,results))

        #sort class probability value in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []

        for i in results:
            return_list.append((self.classes[i[0]],str(i[1])))
        return return_list
    
    def results(self,sentence,userID):
        #if context is maintained then filter class(tag) accordingly
        if sentence.isdecimal():
            if context[userID]=="historydetails":
                return self.classify('ordernumber')
        return self.classify(sentence)
    
    def response(self,sentence,userID='TechVidvan'):
        #get class of users query
        results=self.results(sentence,userID)
        print(sentence,results)
        #store random response to the query
        ans=""
        if results:
            while results:
                for i in self.intents['intents']:
                    #check if tag == query's class
                    if i['tag'] == results[0][0]:

                        #if class contains key as "set"
                        #then store key as userid along with its value in
                        #context dictionary
                        if 'set' in i and not 'filter' in i:
                            context[userID] = i['set']
                        #if the tag doesn't have any filter return response
                        if not 'filter' in i:
                            ans=random.choice(i['responses'])
                            print("Query:",sentence)
                            print("Bot:",ans)

                        #if a class has key as filter then check if context dictionary key's value is same as filter value
                        #return the random response
                        if userID in context and 'filter' in i and i['filter']==context[userID]:
                            if 'set' in i:
                                context[userID] = i['set']
                            ans=random.choice(i['responses'])
                            
                results.pop(0)
        #if ans contains some value then return response to user's query else return some message
        return ans if ans!="" else "Sorry ! I am still Learning.\nYou can train me by providing more datas."

from tkinter import *
from tkinter import ttk
import json
#import the training.py
#and testing.py file
import testing as testpy
import training as trainpy

BG_GRAY="#ABB2B9"
BG_COLOR="#000"
TEXT_COLOR="#FFF"
FONT="Helvetica 14"
FONT_BOLD="Helvetica 13 bold"

class ChatBot:
    def __init__(self):
        #initialize tkinter window
        self.window=Tk()
        self.main_window()
        self.test=testpy.Testing()
        
    #run window
    def run(self):
        self.window.mainloop()
    
    def main_window(self):
        #add title to window and configure it
        self.window.title("ChatBot")
        self.window.resizable(width=False,height=False)
        self.window.configure(width=520,height=520,bg=BG_COLOR)
        #add tab for Chatbot and Train Bot in Notebook frame
        self.tab = ttk.Notebook(self.window)
        self.tab.pack(expand=1,fill='both')
        self.bot_frame=ttk.Frame(self.tab,width=520,height=520)
        self.train_frame=ttk.Frame(self.tab,width=520,height=520)
        self.tab.add(self.bot_frame,text='JARVIS'.center(100))
        self.tab.add(self.train_frame,text='Train Bot'.center(100))
        self.add_bot()
        self.add_train()
        
    def add_bot(self):
        #Add heading to the Chabot window
        head_label=Label(self.bot_frame,bg=BG_COLOR,fg=TEXT_COLOR,text="Welcome to Stark Industries",font=FONT_BOLD,pady=10)
        head_label.place(relwidth=1)
        line = Label(self.bot_frame,width=450,bg=BG_COLOR)
        line.place(relwidth=1,rely=0.07,relheight=0.012)

        #create text widget where conversation will be displayed
        self.text_widget=Text(self.bot_frame,width=20,height=2,bg="#fff",fg="#000",font=FONT,padx=5,pady=5)
        self.text_widget.place(relheight=0.745,relwidth=1,rely=0.08)
        self.text_widget.configure(cursor="arrow",state=DISABLED)

        #create scrollbar
        scrollbar=Scrollbar(self.text_widget)
        scrollbar.place(relheight=1,relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        #create bottom label where message widget will placed
        bottom_label=Label(self.bot_frame,bg=BG_GRAY,height=80)
        bottom_label.place(relwidth=1,rely=0.825)
        #this is for user to put query
        self.msg_entry=Entry(bottom_label,bg="#2C3E50",fg=TEXT_COLOR,font=FONT)
        self.msg_entry.place(relwidth=0.788,relheight=0.06,rely=0.008,relx=0.008)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>",self.on_enter)
        #send button which will call on_enter function to send the query
        send_button=Button(bottom_label,text="Send",font=FONT_BOLD,width=8,bg="Green",command=lambda: self.on_enter(None))   
        send_button.place(relx=0.80,rely=0.008,relheight=0.06,relwidth=0.20)

    def add_train(self):
        #Add heading to the Train Bot window
        head_label=Label(self.train_frame,bg=BG_COLOR,fg=TEXT_COLOR,text="Train Bot",font=FONT_BOLD,pady=10)
        head_label.place(relwidth=1)

        #Tag Label and Entry for intents tag. 
        taglabel=Label(self.train_frame,fg="#000",text="Tag",font=FONT)
        taglabel.place(relwidth=0.2,rely=0.14,relx=0.008)
        self.tag=Entry(self.train_frame,bg="#fff",fg="#000",font=FONT)
        self.tag.place(relwidth=0.7,relheight=0.06,rely=0.14,relx=0.22)

        #Pattern Label and Entry for pattern to our tag.
        self.pattern=[]
        for i in range(2):
            patternlabel=Label(self.train_frame,fg="#000",text="Pattern%d"%(i+1),font=FONT)
            patternlabel.place(relwidth=0.2,rely=0.28+0.08*i,relx=0.008)
            self.pattern.append(Entry(self.train_frame,bg="#fff",fg="#000",font=FONT))
            self.pattern[i].place(relwidth=0.7,relheight=0.06,rely=0.28+0.08*i,relx=0.22)

        #Response Label and Entry for response to our pattern.
        self.response=[]
        for i in range(2):
            responselabel=Label(self.train_frame,fg="#000",text="Response%d"%(i+1),font=FONT)
            responselabel.place(relwidth=0.2,rely=0.50+0.08*i,relx=0.008)
            self.response.append(Entry(self.train_frame,bg="#fff",fg="#000",font=FONT))
            self.response[i].place(relwidth=0.7,relheight=0.06,rely=0.50+0.08*i,relx=0.22)

        #to train our bot create Train Bot button which will call on_train function
        bottom_label=Label(self.train_frame,bg=BG_GRAY,height=80)
        bottom_label.place(relwidth=1,rely=0.825)

        train_button=Button(bottom_label,text="Train Bot",font=FONT_BOLD,width=12,bg="Green",command=lambda: self.on_train(None))
        train_button.place(relx=0.20,rely=0.008,relheight=0.06,relwidth=0.60)
    
    def on_train(self,event):
        #read intent file and append created tag,pattern and responses from add_train function
        with open('C:/Users/user/Downloads/chatbot-code/chatbot-code/intents.json','r+') as json_file:
            file_data=json.load(json_file)
            file_data['intents'].append({
            "tag": self.tag.get(),
            "patterns": [i.get() for i in self.pattern],
            "responses": [i.get() for i in self.response],
            "context": ""
            })
            json_file.seek(0)
            json.dump(file_data, json_file, indent = 1)
        #run and compile model from our training.py file.
        train=trainpy.Training()
        train.build(); print("Trained Successfully")
        self.test=testpy.Testing()
        
    def on_enter(self,event):
        #get user query and bot response
        msg=self.msg_entry.get()
        self.my_msg(msg,"You")
        self.bot_response(msg,"Bot")
        
    def bot_response(self,msg,sender):
        self.text_widget.configure(state=NORMAL)
        #get the response for the user's query from testing.py file
        self.text_widget.insert(END,str(sender)+" : "+self.test.response(msg)+"\n\n")
        self.text_widget.configure(state=DISABLED)
        self.text_widget.see(END)
    
    def my_msg(self,msg,sender):
        #it will display user query and bot response in text_widget
        if not msg:
            return
        self.msg_entry.delete(0,END)
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END,str(sender)+" : "+str(msg)+"\n")
        self.text_widget.configure(state=DISABLED)
        
# run the file
if __name__=="__main__":
    bot = ChatBot()
    bot.run()
