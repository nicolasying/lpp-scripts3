
#
# Note: This Makefile is not clever, i.e. it  does not know about dependencies
# Time-stamp: <2018-04-08 14:43:15 cp983411>

SHELL := /bin/bash

export PATH := bin:$(PATH)
export PYTHONPATH := bin:$(PYTHONPATH)

# DO NOT EDIT THIS PART (unless you know what you are doing)
# set the environment variables in the shell before calling `make`
export LINGUA ?= fr
export REGS ?=  freq rms wrate cwrate sim_dim634_voc56665
export MODEL ?= rms-wrate-cwrate-freq-sim
export MODEL_DIR ?= models/$(LINGUA)/$(MODEL)

export SUBJECTS_FMRI_DATA ?= $(ROOT_DIR)/fmri-data/$(LINGUA)
export ONSETS_DIR ?= inputs/onsets/$(LINGUA)
export REGS_DIR ?= outputs/regressors/$(LINGUA)
export DESIGN_MATRICES_DIR = outputs/design-matrices/$(LINGUA)/$(MODEL)
export FIRSTLEVEL_RESULTS ?= outputs/results-indiv/$(LINGUA)/$(MODEL)
export FIRSTLEVEL_RIDGE_RESULTS ?= outputs/results-indiv-ridge/$(LINGUA)/$(MODEL)
export GROUP_RESULTS ?= outputs/results-group/$(LINGUA)/$(MODEL)
export GROUP_RIDGE_RESULTS ?= outputs/results-group-ridge/$(LINGUA)/$(MODEL)
export ROI_RESULTS ?= outputs/results-group/$(LINGUA)/$(MODEL)-roi
	
regressors:
	mkdir -p $(REGS_DIR)
	python generate-regressors.py --lingua $(LINGUA) --no-overwrite --input-dir $(ONSETS_DIR) --output-dir $(REGS_DIR) $(REGS) 
#	python generate-regressors.py --lingua $(LINGUA) --overwrite --input-dir $(ONSETS_DIR) --output-dir $(REGS_DIR) freq 

design-matrices:
	mkdir -p $(DESIGN_MATRICES_DIR); \
	python merge-regressors.py -i $(REGS_DIR) -o $(DESIGN_MATRICES_DIR) --no-overwrite $(REGS)
	if [ -f $(MODEL_DIR)/orthonormalize.py ]; then \
		echo 'Orthogonalizing...'; \
		python $(MODEL_DIR)/orthonormalize.py \
			--design_matrices=$(DESIGN_MATRICES_DIR) \
			--output_dir=$(DESIGN_MATRICES_DIR); \
	fi
	for f in $(DESIGN_MATRICES_DIR)/*.csv; do python check-design-matrices.py $$f >$${f%.csv}_diagnostics.txt; done

first-level:
	mkdir -p $(FIRSTLEVEL_RESULTS); \
	python $(MODEL_DIR)/firstlevel.py \
		--subject_fmri_data=$(SUBJECTS_FMRI_DATA) \
		--design_matrices=$(DESIGN_MATRICES_DIR) \
		--output_dir=$(FIRSTLEVEL_RESULTS)

first-level-ridge:
	mkdir -p $(FIRSTLEVEL_RIDGE_RESULTS); \
	python models/r2maps_Ridge_nestedcrossval.py \
		--subject_fmri_data=$(SUBJECTS_FMRI_DATA) \
		--design_matrices=$(DESIGN_MATRICES_DIR) \
		--output_dir=$(FIRSTLEVEL_RIDGE_RESULTS) \
		--model_name=$(MODEL)

second-level:
	mkdir -p $(GROUP_RESULTS); \
	python $(MODEL_DIR)/group.py \
		--data_dir=${FIRSTLEVEL_RESULTS} \
		--output_dir=$(GROUP_RESULTS) 

second-level-ridge:
	mkdir -p $(GROUP_RIDGE_RESULTS); \
	python $(MODEL_DIR)/group_ridge.py \
		--data_dir=${FIRSTLEVEL_RIDGE_RESULTS} \
		--output_dir=$(GROUP_RIDGE_RESULTS) 

roi-analyses:
	python lpp-rois.py --data_dir=${FIRSTLEVEL_RESULTS} --output=$(MODEL)-rois.csv
