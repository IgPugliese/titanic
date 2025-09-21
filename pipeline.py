from sklearn.pipeline import PredictPipeline

class Pipeline:
    
    def __init__(self, steps, predicting_step):
        predicting_step_index = next((i for i, step in enumerate(steps) if predicting_step in step), None)
        
        if predicting_step == None:
            raise Exception("The predicting step was not found") 
    
        self.predicting_pipeliene = PredictPipeline(steps[:predicting_step_index+1])
        self.post_processing_pipeline = steps[predicting_step_index+1:]

        pass
    
    def fit(self, X, y=None, **fit_params):
        return self.predicting_pipeliene.fit(X, y)
    
    def predict(self, X, **predict_params):
        prdiction = self.predicting_pipeliene.fit(X, y)
        partial_post_pro = prdiction
        for postPros in self.post_processing_pipeline:
            partial_post_pro = postPros[1].process(partial_post_pro)
        return partial_post_pro
    
    def transform(self, X):
        return self.predicting_pipeliene.transform(X)
        pass

    def score(self, X, y, sample_weight=None):
        """Apply transforms, and score with the final estimator"""
        pass