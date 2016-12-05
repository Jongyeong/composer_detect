# composer_detectection

<b> Composer Detection based on Harmonic motif </b> <br>
It is a code for my graduation thesis. And it is now in progress.

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
adaptive learning factor in range(0.4 ~ 0.01) - change based on cost function(crossentropy)<br>

<b> Feature selection </b> <br>
It is based on t-Test.<br>
if output is 0, it means that feature is meaningless to distinguish two datas of input.<br>
if output is 1, it means that feaure is meaningful.



<b> It is in progress, not completed </b> <br>
there are some problems of optimaztion in layer more than 9.<br>
Error rate<br>
Two composer(Beethoven, Bach) : 17.8%(min) <br>
Three composer( above two and chopin) : 38% (min)
