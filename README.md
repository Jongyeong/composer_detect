# composer_detection

<b> I recommend to see poster or pptx.</b> <br>

<b> Composer Detection based on Harmonic motif </b> <br>
It is a code for my graduation thesis.

<b> Basic Concept </b><br>
Score -> chord feature extract -> data (in json) : feature_extract.py<br>
verify whether input data is meaningful or not : feature_selection.py<br>
Data -> MLP -> test : MLP_ex.py<br>


<b> Feature extractaction </b><br>
It is based on lib 'music21'<br>
Score -> divide into groups of 2 measures(motif candidate) -> use correlation to verify whether each set is distinct measures or not. 
* I use the coefficient of threshold as criterion.
-> Chordify each motif -> extract feature.

<b> Multilayer Perceptron </b><br>
It is based on numpy <br>
I use <br>
softmax function and logistic (= 1 / (1+exp(-x)).<br>
adaptive learning factor in range(0.01 ~ some value) - change based on cost function(crossentropy) in log2 scale<br>

<b> Feature selection </b> <br>
It is based on t-Test.<br>
if output is 0, it means that feature is meaningless to distinguish two data of input.<br>
if output is 1, it means that feature is meaningful.



<b> It is in progress, not completed </b> <br>
there are some problems of optimization in layer more than 9.<br>
Error rate<br>
Two composers(Beethoven, Bach) : 19.83(min) <br>
Three composers(above two and Chopin) : 37.28% (min)
Four composers (above three and Alkan) : 50.17% (min)
