FROM centos

RUN yum install python3 -y

RUN yum install sudo -y

RUN yum install python3-pip -y
RUN yum install python3-setuptools
RUN pip3 install numpy
RUN pip3 install scipy
RUN yum install git -y
RUN pip3 install scipy
RUN pip3 install sklearn
RUN pip3 install pandas
RUN pip3 install scikit-learn
