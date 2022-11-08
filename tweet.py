# %% [markdown]
# # Phân tích cảm xúc (happy, sad) trên 1 tập tweet bằng logistic regression

# %%
import re
import string
import nltk
import numpy as np
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer

# %% [markdown]
# ### Download Dataset

# %%
# Tải về tập dữ liệu tweets
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# Chia thành 2 tập train và test
# train: 4000 samples, test: 1000 samples
train_pos = all_positive_tweets[:4000]
test_pos = all_positive_tweets[4000:]

train_neg = all_negative_tweets[:4000]
test_neg = all_negative_tweets[4000:]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# Tạo nhãn negative: 0, positive: 1
train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

# %% [markdown]
# ### Tiền xử lý dữ liệu cho tập Tweets

# %%
def basic_preprocess(text):
    '''
    Args:
        text: câu đầu vào
    Output:
        text_clean: danh sách các từ (token) sau khi chuyển sang chữ thường và
            được phân tách bởi khoảng trắng
    '''
    # Bỏ RT
    text = re.sub(r'^RT[\s]+', '', text)
    # Bỏ URL
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # Bỏ hashtag
    text = re.sub(r'#', '', text)
    
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    text_tokens = tokenizer.tokenize(text)
    
     
    text_clean = []
    
    for word in text_tokens:
        if word not in string.punctuation:
            text_clean.append(word)
    
    return text_clean
   

# %%
# Kết quả đầu ra
example_sentence = "RT @Twitter @chapagain Hello There! Have a great day. #good #morning http://chapagain.com.np"
basic_preprocess(example_sentence)

# %%
def count_freq_words(corpus, labels):
    """ Xây dựng bộ từ điển tần suất xuất hiện của các từ
    Args:
        corpus: tập danh sách các câu
        labels: tập nhãn tương ứng với các câu trong corpus (0 hoặc 1)
    Output:
        model: bộ từ điển ánh xạ mỗi từ và tần suất xuất hiện của từ đó trong corpus
            key: (word, label)
            value: frequency
            VD: {('boring', 0): 2} => từ boring xuất hiện 2 lần trong các sample thuộc class 0
    """
    model = {}
    for label, sentence in zip(labels, corpus):
        for word in basic_preprocess(sentence):
            pair = (word, label)
            if pair in model:
                model[pair] +=1
                
            else:
                model[pair]=1
    return model

# %%
freqs = count_freq_words(train_x, train_y)
freqs

# %%
def lookup(freqs, word, label):
    '''
    Args:
        freqs: a dictionary with the frequency of each pair
        word: the word to look up
        label: the label corresponding to the word
    Output:
        count: the number of times the word with its corresponding label appears.
    '''
    count = 0

    pair = (word, label)
    if pair in freqs:
        count = freqs[pair]

    return count

lookup(freqs, "happy", 0), lookup(freqs, "happy", 1)

# %% [markdown]
# ### Trích xuất các feature
# Chuyển từ `tweet` sang feature
# Với mỗi `tweet` sẽ được biểu diễn bởi 2 feature:
# - số lượng các positive words
# - số lượng các negative words

# %%
def extract_features(text, freqs):
    '''
    Args: 
        text: tweet
        freqs: bộ từ điển tần suất xuất hiện của từ theo label (word, label)
    Output: 
        x: vector feature có chiều (1,3)
    '''
    # tiền xử lý
    word_l = basic_preprocess(text)
    
    # 3 thành phần: bias, feature 1 và feature 2
    x = np.zeros((1, 3)) 
    
    # bias
    x[0,0] = 1 
    
    ### START CODE HERE
    for word in word_l:
        x[0,1]+=lookup(freqs, word, 1)
        x[0,2]+=lookup(freqs, word, 0)
        
    ### END CODE HERE ###
    assert(x.shape == (1, 3))
    return x

# %%
# Kiểm tra
freqs = count_freq_words(train_x, train_y)
print(train_x[0])
extract_features(train_x[0], freqs)

# %% [markdown]
# ### Logistic Regression

# %% [markdown]
# #### Sigmoid
# The sigmoid function: 
# 
# $$ h(z) = \frac{1}{1+\exp^{-z}} \tag{1}$$
# 

# %%
def sigmoid(z): 
    '''
    Args:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    
    h = 1 / (1 + np.exp(-z))
    
    return h

# %%
# Kết quả kiểm tra hàm sigmpoid
sigmoid(0) == 0.5, sigmoid(4.92) == 0.9927537604041685

# %% [markdown]
# #### Gradient Descent Function
# * Số vòng lặp huấn luyện mô hình: `num_iters`
# * Với mỗi vòng lặp chúng ta sẽ tính `logits-z`, cost và cập nhật trọng số
# * Số samples training: `m`, số features trên mỗi sample: `n`
# * Trọng số mô hình:  
# $$\mathbf{\theta} = \begin{pmatrix}
# \theta_0
# \\
# \theta_1
# \\ 
# \theta_2 
# \\ 
# \vdots
# \\ 
# \theta_n
# \end{pmatrix}$$
# 
# * Tính `logits-z`:   $$z = \mathbf{x}\mathbf{\theta}$$
#     * $\mathbf{x}$ có chiều (m, n+1) 
#     * $\mathbf{\theta}$: có chiều (n+1, 1)
#     * $\mathbf{z}$: có chiều (m, 1)
# * Dự đoán y_hat có chiều (m,1):$$\widehat{y}(z) = sigmoid(z)$$
# * Lost function $J$:
# $$J = \frac{-1}{m} \times \left(\mathbf{y}^T \cdot log(\mathbf{h}) + \mathbf{(1-y)}^T \cdot log(\mathbf{1-h}) \right)$$
# * Cập nhật `theta`:
# $$\mathbf{\theta} = \mathbf{\theta} - \frac{\alpha}{m} \times \left( \mathbf{x}^T \cdot \left( \mathbf{\widehat{y}-y} \right) \right)$$

# %%
def gradient_descent(x, y, theta, alpha, num_iters):
    '''
    Args:
        x: matrix of features, có chiều (m,n+1)
        y: label tương ứng (m,1)
        theta: vector trọng số (n+1,1)
        alpha: tốc độ học
        num_iters: số vòng lặp
    Output:
        J: final cost
        theta: vector trọng số
    '''
    m = len(x)
    
    for i in range(0, num_iters):
        z=np.dot(x, theta)
        y_hat=sigmoid(z)
        J=(-1/m) * (np.dot(y.T, np.log(y_hat))+np.dot((1-y).T, np.log(1-y_hat)))
        theta=theta-(alpha/m)*(np.dot(x.T, (y_hat-y)))

    return J, theta

# %%
# Kiểm tra
# freqs tương tự mục 1.2
# VD: các từ không có trong bộ `freq`
x_test = "việt nam"
extract_features(x_test, freqs)

# %% [markdown]
# ### Huấn luyện mô hình Logistic Regression

# %%
# Tạo ma trận X có kích thước mxn với m là số sample, n=3 (số features + bias)
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

Y = np.expand_dims(train_y, 1)

# Huấn luyện với số vòng lặp 1500, tốc độ học 1e-6
J, theta = gradient_descent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"Cost {J.item()}.")
print(f"Weight {theta}")

# %% [markdown]
# ### Dự đoán
# * Tiền xử lý với dữ liệu thử nghiệm
# * Tính `logits` dựa vào công thức
# 
# $$y_{pred} = sigmoid(\mathbf{x} \cdot \theta)$$

# %%
# Ex 10
def predict_tweet(text, freqs, theta):
    '''
    Args: 
        text: tweet
        freqs: bộ từ điển tần suất xuất hiện của từ theo label (word, label)
        theta: (3,1) vector trọng số
    Output: 
        y_pred: xác suất dự đoán
    '''
    x = extract_features(text, freqs)
    y_pred = sigmoid(np.dot(x, theta))
    
    return y_pred

# %%
tests = ["happy", "sad"]
for t in tests:
    pred = predict_tweet(t, freqs, theta)
    print(f'{t} -> {pred}')

# %%
predict_tweet("I'm very happy", freqs, theta)

# %% [markdown]
# ### Đánh giá độ chính xác trên tập test

# %%
acc = 0
for sentence, label in zip(test_x, test_y):

    # predic each sentence in test set
    pred = predict_tweet(sentence, freqs, theta)

    if pred > 0.5:
        pred_l = 1
    else:
        pred_l = 0

    # compare predict label with target label
    if int(pred_l) == int(label):
        acc += 1

print('Accuracy: ', acc/len(test_x))

# %%



