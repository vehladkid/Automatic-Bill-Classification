# HR Bill Classifier - 5 to 7 minute spoken script
(About 900 words at a normal pace. Numbers are from the repository; details and sources are in `EXAMINER_PROJECT_WALKTHROUGH.md`. Keep the app open at http://localhost:8501 for the demo.)

## 1. Problem (0:00 - 0:40)
"HR receives expense claims as photographs or PDFs of bills. Checking each one by hand is slow. My project reads a bill and answers two questions. Task A: what was the money spent on - food, hotel, fuel, electricity and so on. Task B: what are the key details on the bill - vendor, invoice number, date, GSTIN, totals and taxes. If the system is not confident, it sends the bill to a human instead of approving it automatically."

## 2. Dataset (0:40 - 1:40)
"I built the dataset myself. It has 8,183 documents: 5,753 real bills and 2,430 synthetic Indian bills. Real sources include SROIE, CORD, ReceiptBench, my own collected Indian bills, utility bills and a grocery-GST set. Every document has provenance, licence, country, language and label-quality fields in one manifest.
Synthetic bills fill gaps where no real labelled data exists, for example rent and HSN codes. But I never report synthetic scores as results, because they are rendered from a few templates and every model scores nearly 100 percent on them. All headline numbers are on real bills only.
To avoid cheating by accident, similar bills are grouped: exact and perceptual-hash duplicates and same-merchant families always stay on one side of the train, validation and test split. An independent audit found no validation or test image within a close perceptual distance of a training image. One honest gap: there are no real rent documents."

## 3. Architecture (1:40 - 2:20)
"The flow is: upload a bill, OCR reads the text and word positions, Task A predicts the category, Task B extracts the fields, the interface shows everything, and the original bill is saved in a folder named after its predicted category. OCR is a shared step - it is not a separate task. Everything runs locally on one laptop GPU."

## 4. Task A (2:20 - 3:30)
"Task A uses three models and combines them. A text model - TF-IDF with logistic regression - reads the OCR words. A ConvNeXt image model looks at the picture. A LayoutLMv3 model uses text, layout and image together. Each model's probabilities are calibrated with a temperature, then combined as a weighted average, and a confidence threshold of 0.61 decides whether a bill is processed automatically or goes to HR.
On real test bills the combined model reaches 90.7 percent accuracy and 83.9 percent macro-F1; on real Indian bills 95.5 percent, from 110 documents. I report macro-F1 as well, because accuracy is dominated by big classes like food, while macro-F1 shows weak ones: medical is about 0.50 and bus and water about 0.73, and several classes have fewer than ten real test bills. The threshold sends about 7 percent of real test bills to HR and the rest are about 94 percent correct."

## 5. Task B (3:30 - 4:50)
"Task B extracts 34 possible fields. There are three extractors. A rule-based baseline using patterns. A candidate ranker that lists possible values and scores them with gradient boosting. And a LayoutLMv3 token classifier that tags each word, using begin, inside and outside tags, as part of a field or not. For each field I choose the best of the three using real validation data only, then calibrate a confidence score so it means the probability of being right.
On the frozen real test set the rules reach 0.44 micro-F1, the ranker 0.55, LayoutLMv3 0.65 and the combined system 0.66. Totals and tax totals are around 0.82; merchant name is 0.54 and invoice number 0.42, so those need a human check. The Indian tax fields rest on only a handful of real test bills, so I do not quote them as accuracies. Honestly, this is a research prototype, not a production system."

## 6. Application and a live example (4:50 - 5:50)
"The application is a local web app. You upload a JPG, PNG or PDF; it validates the file, runs OCR once, runs Task A and Task B, shows the original, the OCR text, the category with its confidence, and a table of extracted fields, and saves the bill under outputs/classified in its category folder. If Task B fails you still get the OCR and category. Files get unique names and are never overwritten. I tested this with 26 automated checks covering saving, duplicates, bad files and simulated failures.
Here is a real restaurant bill I tested. It was classified as food after I fixed an integration problem that had made it miscellaneous. Task B got the invoice number, GSTIN and total right, but picked the wrong vendor line and returned a single CGST line when the bill has two. That one bill is an observation, not a benchmark."

## 7. QA status, limitations, conclusion (5:50 - 6:40)
"I also started a second version of Task B using real Indian GST invoices, with multiple HSN codes and extra tax fields. Fifty-four invoices were annotated, but by an AI annotator, so I treat them as unverified. An independent human check is required and has not been done - the answer sheet is still blank - so that version is blocked, and I have not retrained anything.
Main limitations: no real rent data, small real test sets for some classes, thin Indian tax-field evidence, and Task B is unreliable for vendor and invoice number. In conclusion: Task A is a solid, calibrated classifier with honest limits; Task B is a working but modest extractor; and the application ties them into a working demo. The next step is human verification of the new annotations and more real Indian invoices."
