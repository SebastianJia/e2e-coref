from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import metrics
import tensorflow as tf
import coref_model as cm
import util

if __name__ == "__main__":
  config = util.initialize_from_env()

  # Input file in .jsonlines format.
  input_filename = sys.argv[2]

  # Predictions will be written to this file in .jsonlines format.
  output_filename = sys.argv[3]
  coref_predictions = {}
  coref_evaluator = metrics.CorefEvaluator()
  model = cm.CorefModel(config)
  model.config["conll_eval_path"] = sys.argv[4]
  with tf.Session() as session:
    model.restore(session)

    with open(output_filename, "w") as output_file:
      with open(input_filename) as input_file:
        for example_num, line in enumerate(input_file.readlines()):
          example = json.loads(line)
          tensorized_example = model.tensorize_example(example, is_training=False)
          feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
          _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)
          predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
          example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
          coref_predictions[example["doc_key"]] = model.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"], coref_evaluator)

          output_file.write(json.dumps(example))
          output_file.write("\n")
          if example_num % 100 == 0:
            print("Decoded {} examples.".format(example_num + 1))
  summary_dict = {}
  conll_results = conll.evaluate_conll(model.config["conll_eval_path"], coref_predictions, False)
  average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
  summary_dict["Average F1 (conll)"] = average_f1
  print("Average F1 (conll): {:.2f}%".format(average_f1))
  p,r,f = coref_evaluator.get_prf()
  summary_dict["Average F1 (py)"] = f
  print("Average F1 (py): {:.2f}%".format(f * 100))
  summary_dict["Average precision (py)"] = p
  print("Average precision (py): {:.2f}%".format(p * 100))
  summary_dict["Average recall (py)"] = r
  print("Average recall (py): {:.2f}%".format(r * 100))
