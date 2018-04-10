The simple baseline takes the first sentence of the article, cleans it (lowercase, remove punctuation), and outputs it as the summary for that article.

For example, for the article text:

A Louisiana family is fighting to protect its beloved pit bull from a "vicious dogs" ordinance. Joanna Armand started an online petition to protect pit bull Zeus from an uncertain fate. The petition, which asks the village of Moreauville to reverse a "vicious dog" ban against pit bulls and rottweilers, has drawn nearly 40,000 signatures. Zeus is far from a vicious dog, Armand told CNN. He provides love and support for her children, especially daughter O'Hara, who suffers from severe neck problems and uses a Halo brace and a wheelchair. "If anything ever happened to him, I would just shut down," O'Hara told KALB. Zeus acts as a sort of therapy dog for O'Hara, Armand told CNN. He sleeps by O'Hara's side every night; if O'Hara wakes up with seizures, he rouses her mother. The family has asked local officials to consider the special circumstances, only to be told there's nothing they can do, Armand told CNN. The village of Moreauville passed an ordinance in October banning pit bulls and rottweilers as of December 1. Owners of the animals received a letter saying that pets found within the corporate limits of Moreauville after that date will be "impounded and transported to a veterinary clinic for further disposition," according to a letter posted on the family's Facebook page. The mayor and alderman of Moreauville did not respond to CNN's repeated requests for comment. Alderman Penn Lemoine told KALB that the ordinance was created to placate residents in the small village. "We had several residents that were complaining about not being able to walk along the neighborhoods because these dogs were basically running along town," Lemoine said, according to KALB. Moreauville is not the first to pass a ordinance restricting the ownership of pit bulls, rottweilers or other so-called "vicious dogs." But the American Society for the Prevention of Cruelty to Animals recommends breed neutral laws "that focus, not on breed, but on people's responsibility for their dogs' behavior, including measures that hold owners of all breeds accountable for properly housing, supervising and controlling their dogs." The White House also signaled its opposition to breed-specific legislation in 2013, saying "research shows that bans on certain types of dogs are largely ineffective and often a waste of public resources." Several states have passed legislation banning local governments from introducing breed-specific laws or ordinances. Besides, pit bulls are not a breed of dog. Several breeds of dogs, mainly the Bull Terrier, the Bull dog and the American Staffordshire Terrier are commonly referred to as "pit bulls." These dogs descended from an English bull-baiting dog bred to bite and hold bulls, bears and other large animals around the face and head, according to the ASPCA. They were later bred with terriers to produce a more agile, athletic type dog. "It is likely that that the vast majority of pit bull type dogs in our communities today are the result of random breeding -- two dogs being mated without regard to the behavioral traits being passed on to their offspring. The result of random breeding is a population of dogs with a wide range of behavioral predispositions. For this reason it is important to evaluate and treat each dog, no matter its breed, as an individual."

The gold summary text is:

Family starts petition asking village to reverse ban against pit bulls and rottweilers Family fears Zeus will be taken from them pursuant to village's "vicious dogs" ordnance Zeus provides children with love and support, they say

And the baseline would output:

a louisiana family is fighting to protect its beloved pit bull from a "vicious dogs" ordinance

The baseline can be run with python3 simple-baseline.py --goldfile FILE --predfile OUTFILE

ex: python3 simple-baseline.py --goldfile sumdata/bothdev.txt --predfile base_dev_pred.txt

On the test set, when evaluating with bigrams,
Precision: 0.07162823896253027
Recall: 0.03750891711392287
FScore: 0.04923525176667997

And when evaluating with unigrams,
Precision: 0.30527854879188576
Recall: 0.16272275489582794
FScore: 0.21228900893474242
