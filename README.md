# ddsp_simplified

This project was done for the Deep Unsuperised Learning COMP447 Spring'21 course at Koç University by Recep Oğuz Araz raraz15@ku.edu.tr and Haldun Balım hbalim15@ku.edu.tr.

The code here is mostly taken from https://github.com/magenta/ddsp the original DDSP repository.

We implemented the Supervised and Unsupervised DDSP Autoencoders, trained and compared their performances.

The goal was to have an easy to use DDSP implementation that is stripped from the cool but hard to understand parts of the original repository.

You just need the DDSP, librosa and some common python packages to use it.

Training the Supervised and Unsupervised autoencoders can be done using the train_supervised.py and train_unsupervised.py, with providing a config file.

After you have a trained model, you can perform timbre transfer using the timbre_transfer.py file.

Currently the unsupervised part requires more work to be done to implement the architecture described in the paper.

If you are from the Google Magenta team and you are unhappy about the existence of this repository, please contact us and we can solve it together.

