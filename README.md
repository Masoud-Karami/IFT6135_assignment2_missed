# IFT6135 Representation Learning_sequential language models
Word Level Models
  # Penn Treebank
Convolutional Neural Network
----------------------------

Let <img src="/tex/c416d0c6d8ab37f889334e2d1a9863c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.628015599999989pt height=14.611878600000017pt/> <img src="/tex/5ba9e09976f6a5a8919c63baa6f2fbe7.svg?invert_in_darkmode&sanitize=true" align=middle width=10.95894029999999pt height=17.723762100000005pt/> <img src="/tex/76b11a20d53ed4d10c9d38e8b4ecd46a.svg?invert_in_darkmode&sanitize=true" align=middle width=19.13820809999999pt height=27.91243950000002pt/> be the <img src="/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-dimensional word
vector correspond to the <img src="/tex/77a3b857d53fb44e33b53e4c8b68351a.svg?invert_in_darkmode&sanitize=true" align=middle width=5.663225699999989pt height=21.68300969999999pt/>-th word in the input sentence. After
padding all sentences in an input batch to the same length <img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/>, where
<img src="/tex/55a049b8f161ae7cfeb0197d75aff967.svg?invert_in_darkmode&sanitize=true" align=middle width=9.86687624999999pt height=14.15524440000002pt/> is the maximum length sentence of all sentences in the batch, each
sentence is then represented as

<p align="center"><img src="/tex/95e6739128498ae9459d6a100d53fe47.svg?invert_in_darkmode&sanitize=true" align=middle width=177.89524335pt height=12.05477955pt/></p>

where <img src="/tex/45848451c711deba755da6422f9e68c6.svg?invert_in_darkmode&sanitize=true" align=middle width=12.785434199999989pt height=19.1781018pt/> is the concatenation operation. Let <img src="/tex/bfddb4c677ca74c5212b9bdbe4532f68.svg?invert_in_darkmode&sanitize=true" align=middle width=39.19628294999999pt height=14.611878600000017pt/>
represent the concatenation of words <img src="/tex/c416d0c6d8ab37f889334e2d1a9863c3.svg?invert_in_darkmode&sanitize=true" align=middle width=14.628015599999989pt height=14.611878600000017pt/>, <img src="/tex/48daf924d1e550eb78217f1e0884411d.svg?invert_in_darkmode&sanitize=true" align=middle width=31.27193519999999pt height=14.611878600000017pt/>,
..., <img src="/tex/c44f404c5862ec20b77e284ed02e857b.svg?invert_in_darkmode&sanitize=true" align=middle width=30.82389749999999pt height=14.611878600000017pt/>. In Convolutional neural networks, we apply
convolution operations <img src="/tex/5ddc1b22140b2658931d8962d8c90c33.svg?invert_in_darkmode&sanitize=true" align=middle width=13.91546639999999pt height=14.611878600000017pt/> <img src="/tex/5ba9e09976f6a5a8919c63baa6f2fbe7.svg?invert_in_darkmode&sanitize=true" align=middle width=10.95894029999999pt height=17.723762100000005pt/> <img src="/tex/2c5a948318138412ea3a0dec0a6d7290.svg?invert_in_darkmode&sanitize=true" align=middle width=30.738368099999988pt height=27.91243950000002pt/> with filter
size <img src="/tex/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode&sanitize=true" align=middle width=9.47111549999999pt height=22.831056599999986pt/> to produce features, where the filter size is effectively the
window size of words to convolve over. Let <img src="/tex/3bc6fc8b86b6c61889f4e572c7546b8e.svg?invert_in_darkmode&sanitize=true" align=middle width=11.76470294999999pt height=14.15524440000002pt/> be a feature generated
by this operation. Then

<p align="center"><img src="/tex/0ce891ff1a4f93f91f68cf06f3da3be3.svg?invert_in_darkmode&sanitize=true" align=middle width=168.47671499999998pt height=16.438356pt/></p>

where <img src="/tex/4bdc8d9bcfb35e1c9bfb51fc69687dfc.svg?invert_in_darkmode&sanitize=true" align=middle width=7.054796099999991pt height=22.831056599999986pt/> is a bias term and <img src="/tex/190083ef7a1625fbc75f243cffb9c96d.svg?invert_in_darkmode&sanitize=true" align=middle width=9.81741584999999pt height=22.831056599999986pt/> is the rectified linear unit (ReLU)
function. Applying filter length size <img src="/tex/31fae8b8b78ebe01cbfbe2fe53832624.svg?invert_in_darkmode&sanitize=true" align=middle width=12.210846449999991pt height=14.15524440000002pt/> over all possible windows of
the words in our input sentence produces the feature map

<p align="center"><img src="/tex/8d71ca3d40ca59ad73cf8f5faad703c3.svg?invert_in_darkmode&sanitize=true" align=middle width=165.36159585pt height=16.438356pt/></p>

In our implementation, we convolve over filter sizes <img src="/tex/2ad9d098b937e46f9f58968551adac57.svg?invert_in_darkmode&sanitize=true" align=middle width=9.47111549999999pt height=22.831056599999986pt/> <img src="/tex/5ba9e09976f6a5a8919c63baa6f2fbe7.svg?invert_in_darkmode&sanitize=true" align=middle width=10.95894029999999pt height=17.723762100000005pt/>
<img src="/tex/deab25011f1573e316f68b62accb9567.svg?invert_in_darkmode&sanitize=true" align=middle width=55.70780654999999pt height=24.65753399999998pt/> and then concatenate the features of each
<img src="/tex/f5ec0198af7987f6245d92311996f877.svg?invert_in_darkmode&sanitize=true" align=middle width=16.09780754999999pt height=14.611878600000017pt/> into a single vector. We apply a max-over-time pooling
operation (Collobert et al, 2011) to this vector of concatenated feature
maps, denoted <img src="/tex/e74308ca1bd81a80819135589e16d2e6.svg?invert_in_darkmode&sanitize=true" align=middle width=8.40178184999999pt height=14.611878600000017pt/>, and get <img src="/tex/0038bd66465254af4225aa31848b342b.svg?invert_in_darkmode&sanitize=true" align=middle width=8.579777249999989pt height=22.831056599999986pt/> = max(<img src="/tex/e74308ca1bd81a80819135589e16d2e6.svg?invert_in_darkmode&sanitize=true" align=middle width=8.40178184999999pt height=14.611878600000017pt/>). We
then apply dropout with <img src="/tex/2ec6e630f199f589a2402fdf3e0289d5.svg?invert_in_darkmode&sanitize=true" align=middle width=8.270567249999992pt height=14.15524440000002pt/> = 0.50 to <img src="/tex/0038bd66465254af4225aa31848b342b.svg?invert_in_darkmode&sanitize=true" align=middle width=8.579777249999989pt height=22.831056599999986pt/> as regularization
measure against overfitting, pass this into a fully-connected layer and
compute the softmax over the output.

Modified CNN
------------

Finally, we also implemented with a series of modifications to the CNN
architecture to give a slight performance improvement on the SST-2
dataset. In this implementation, we utilize Stanford’s GloVe pre-trained
vectors (Pennington et al., 2014), we make these changes:

-   Following (Kim, 2014), we use two copies of word embedding table
    during the convolution and max-pooling steps – one that is
    non-static, or updated during training as a regular module in the
    model, and another that is omitted from the optimizer and preserved
    as static throughout the training run. In the forward pass of the
    model, these two sets of embeddings are concatenated together along
    the “channel” dimension, and then passed into the three
    convolutional layers as a single tensor, with two values for each of
    the 300 dimensions in GloVe model.

-   After producing the combined feature vector representing the
    max-pooled features from the three convolutional kernels, we simply
    add the non-padded word count of the input as a single extra
    dimension, producing a 301-dimension tensor which then gets mapped
    to the 2-unit output. From an engineering standpoint, we find that
    this marginally improves performance on the SST-2 dataset, where, on
    average, positive sentences are slightly longer than negative ones –
    19.41 words versus 19.17. It’s not clear whether this would hold
    across different data sets, or if it’s specific to SST-2. (Though
    it’s also not entirely clear that wouldn’t, and seems to imply an
    interesting corpus-linguistic question – are “positive” sentences
    generally longer than “negative” ones?)
