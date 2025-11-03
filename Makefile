REPO=nail04.ece.ucsb.edu:5000/
CONT=nail04.ece.ucsb.edu:5000/bq3-test:v0.01


all: .pkg Dockerfile.stable.caffe.xenial
	docker build -t $(CONT) -f Dockerfile.stable.caffe.xenial .
	docker push $(CONT)

.pkg:
#	if [ ! -d connoisseur ] ; then hg clone ssh://hg@bitbucket.org/dimin/connoisseur; fi



publish:
	if [[ ! -z "$(REPO)" ]] ; then \
		docker tag $(CONT) $(REPO)$(CONT);\
        docker push $(REPO)$(CONT); \
    fi


# nvidia-docker run -it --rm -p 8111:80 bisque_ucsb_caffe:dev bootstrap bash
#
