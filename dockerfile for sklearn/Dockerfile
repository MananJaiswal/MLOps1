FROM centos

RUN yum install python3 -y

RUN yum install virtualenv -y

RUN pip3 --no-cache-dir install numpy

RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install statsmodels
RUN pip3 install matplotlib
RUN pip3 install Pillow 
RUN pip3  install papermill

RUN yum install python3-pip -y
RUN yum install python3-setuptools -y
RUN yum install python3-wheel -y
RUN yum install pkg-config -y

RUN yum install sudo -y
RUN pip3 --no-cache-dir install
RUN pip3 install --upgrade setuptools
RUN pip3 install ez_setup
RUN pip3 install --upgrade pip

RUN pip3  install tensorflow==1.12.0
RUN pip3 install --upgrade tensorflow-probability
RUN pip3  install keras==2.2.4
RUN yum install git -y

COPY . /braindirectory
WORKDIR /braindirectory
ENTRYPOINT ["python"]
CMD [". /BT_Detection.py"]



