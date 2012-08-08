(ns entendre.core
  (:use [clojure-encog.nnets]
        [clojure-encog.training]
        [clojure-encog.normalization])
  (:import [org.encog.ml.train.strategy RequiredImprovementStrategy]))

(defn -main []
  (let [network (make-network {:input   2
                               :output  1
                               :hidden [50 10]}
                              (make-activationF :sigmoid)
                              (make-pattern     :feed-forward))
        xor-input [[0.0 0.0] [1.0 0.0] [0.0 0.1] [1.0 1.0]]
        xor-ideal [[0.0] [1.0] [1.0] [0.0]]
        dataset   (make-data :basic-dataset xor-input xor-ideal)
        trainer   ((make-trainer :back-prop) network dataset)]
    (train trainer 0.01 500 [(RequiredImprovementStrategy. 5)])))
