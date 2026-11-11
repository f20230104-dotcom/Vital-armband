"""Three-person speaker script for the 14-slide deck; build_presentation.py writes it into the speaker notes.

Each entry: (speaker number, approximate seconds, spoken text). Replace [Speaker N] with names.
"""

SCRIPT = [
    (1, 20, "Good morning, everyone. We are [Speaker 1], [Speaker 2] and [Speaker 3], and our project is the Vital "
            "Wristband: a low-cost wristband that detects falls and monitors vital signs, with a small neural network "
            "running directly on an ESP32 chip. I'll explain why this matters, [Speaker 2] will show how we built and "
            "tested it, and [Speaker 3] will cover our results and the plan to finish the hardware."),
    (1, 40, "Falls are a bigger problem than most people think. According to the WHO, about 684,000 people die from "
            "falls every year, and 37 million falls are serious enough to need medical care. Often the fall itself "
            "isn't the worst part; it's lying on the floor for hours before anyone notices. So our wristband does four "
            "things: it detects falls on the device, it tracks heart rate, oxygen and breathing, it reminds the wearer "
            "about medication, and it alerts a caregiver through a live dashboard."),
    (1, 35, "Our problem statement is simple: detect falls and abnormal vitals on a wristband that costs about thirty "
            "dollars, without relying on the cloud, and alert a caregiver within seconds. Today's options fall short "
            "in three ways. Smartwatches with fall detection cost three to four hundred dollars. Simple threshold "
            "detectors give too many false alarms. And none of them tell anyone when a medication dose is missed. On "
            "top of that, everything had to fit into a chip with 520 kilobytes of memory, running on a battery."),
    (1, 35, "Here's how existing products compare. Apple, Samsung and Google watches detect falls, but they're "
            "expensive and only track breathing while you sleep. Fitness and budget bands don't detect falls at all, "
            "and medical-alert watches charge a monthly fee. Ours is the only one combining all four features, at "
            "about a tenth of the price, with no subscription."),
    (1, 30, "On the research side, earlier work either used simple thresholds, ran the model on a PC, placed the sensor "
            "on the waist, or used networks too large for a microcontroller. The gap we fill is one wrist device, with "
            "a small model running on the chip itself, tested on people it has never seen.\n\nHANDOVER: Now I'll hand "
            "over to [Speaker 2], who'll explain how we built it."),
    (2, 40, "Thanks, [Speaker 1]. The system has three parts. On the left are the sensors: a motion sensor, an optical "
            "sensor for heart rate and oxygen, a real-time clock, two buttons and the battery. In the middle, the ESP32 "
            "runs four tasks in parallel using FreeRTOS: fall detection, vital signs, medication reminders and WiFi "
            "alerts. On the right, the caregiver's dashboard receives the alerts. As the table shows, all the hardware "
            "costs about thirty dollars."),
    (2, 45, "These are four real scenarios from our dashboard. Top left: a fall is confirmed and the caregiver sees a "
            "red alert. Top right: straight after a fall, the band buzzes for fifteen seconds so the wearer can cancel "
            "a false alarm with one button. Bottom left: a jump looks like a fall at first, but the person keeps "
            "moving, so no alert is sent. Bottom right: blood oxygen stays at 87 percent for thirty seconds, which "
            "triggers a vitals alert."),
    (2, 45, "We trained on three datasets. SisFall is the largest public fall dataset, with 38 people. The UCI dataset "
            "gives real wrist recordings from 17 volunteers, and we kept five of them completely unseen for testing. "
            "We also generated synthetic wrist data ourselves. The model is a small convolutional network, only 13.8 "
            "kilobytes after compression. It doesn't run all the time: a cheap trigger spots an impact, we check that "
            "the person stays still, and only then does the network decide. The wearer then gets 15 seconds to cancel."),
    (2, 45, "We tested this carefully. The chart shows how many falls we caught from the unseen wrist volunteers. "
            "Trained on SisFall alone, the model caught 80 percent. Adding random rotations during training, so the "
            "model doesn't care how the band sits on the wrist, plus real wrist data, raised that to 98 percent. Plain "
            "thresholds fail badly, and a Random Forest is decent but about a hundred times bigger than our model. "
            "Across five rounds of cross-validation, accuracy stayed at 98.9 percent.\n\nHANDOVER: [Speaker 3] will "
            "now take you through the results."),
    (3, 45, "Thanks, [Speaker 2]. On 17 people the model had never seen, it reaches 98.3 percent accuracy, 98.9 percent "
            "precision and 97.9 percent recall, and it does just as well on real wrist data. Testing also caught an "
            "important design flaw. Our original trigger let only 44 percent of real falls reach the model, because "
            "people often grab something or slide instead of falling freely. Adding a second impact trigger raised "
            "that to 99 percent. In an 80-minute simulation of the full firmware, the band caught all 20 falls with "
            "zero false alerts."),
    (3, 35, "Will it run on the real chip? Memory fits easily: the firmware uses about half the flash and a quarter of "
            "the RAM. Speed and battery life are still estimates. We expect the model to run in under 15 milliseconds "
            "and the battery to last 9 to 12 hours, which we need to improve. So the software is ready; what's left "
            "can only be measured on the physical wristband."),
    (3, 35, "Here's our plan to finish by the 10th of December. This week we buy the parts and connect everything on a "
            "breadboard. Next week we record real wrist data and calibrate the sensors. Then we fine-tune the model "
            "and build the case and strap. In the final week we test falls, daily wear and battery life, aiming for at "
            "least 95 percent recall, before the final demonstration."),
    (3, 25, "The main risks are battery life, noisy readings at the wrist, a board that's too large, and limited real "
            "fall data, and each one has a plan. After the course, we'd like to add a phone app, alerts outdoors, "
            "over-the-air updates and, eventually, trials with older adults."),
    (3, 15, "Thank you for listening. You can try the live demo and see all our code at these links. We're happy to "
            "take your questions."),
]
