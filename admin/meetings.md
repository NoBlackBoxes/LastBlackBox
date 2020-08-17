## 17/8/20

**START DATE: SEPTEMBER 14**

in SWC main auditorium, socially distanced

tone -- smart highschooler
accurate mental model, note the caveats

art-style -- novabook II onyx
black on white hand drawing

white paper + black pen + 

look at examples tonight

rpi 4

two rpis
	- one is a robot brain
	- other is your hacking machine

11 students in the SWC course

pi on the robot -- too satisfying not to use this for the forebrain metaphor

NoBlackBoxes license
	- custom license for the project
	- like "Don't Be a Dick License"

vision
	- pi + opencv
	- c + syscalls
	- don't want to rewite vk code


## 16/7/20

### team
- elena
- adam
- spencer
- eirinn
- peter
- ted
- philip
- elina?
- liyuan?
- hugo? 
- ???

### Alan Kay's Error 33
- putting something outside of control into your critical path
- thus PARC built everything from scratch
- goes against "don't reinvent the wheel"
- if you have a group of people who are able to roll their own, you have to
	- which companies have done this? Stripe, FB, etc. 
- e.g. contact tracing app in isle of wight -- black boxes in the critical path, led to failure

### some principles of LBB
- no Error 33's
- cumulative -- but not linear?
- this is a course about brains/intelligence/computation

### homepage == box (digital twin)
- tutorials, videos (vertical)
- e.g. camera
	- other optional things (horizontal)
		- light
		- photons
		- pixels
		- CCDs

### questions
- screen for RPi? small LCD?

### ideas for competition
- stream sensors to Rpi screen, navigate with limited sensor information
- bot is essentially a decorticated animal
- rpi is cortex -- parallel, scheduling
- goals are coordination, hierarchy

### rpi + arduino
- wires via GPIO
- USB
- wireless WiFi

### how do you bridge the gap to Raspbian?
- start with a teletype
- build a teletype...?
- ideally
	- load some compiled code on the SD
	- start the thing up, explain the init binary blob (decompile this?)
	- show something on the screen -- you booted! 

### refs

[nice, simple, js applets](https://probmods.org/)

[microscope](https://hackaday.io/project/11429-internet-of-things-microscope)

[oscilloscope1](https://www.scopefun.com/)

[oscilloscope2](https://www.crowdsupply.com/andy-haas/haasoscope)


nand2tetris covers a lot of the same material — the goal is to understand bits —> computation, then understand compilation, then understand OS — they use their own low- and high-level languages thought

[nand2tetris](https://www.nand2tetris.org/project12)

there’s a lot about programming paradigms that I wish we could cover, but it gets insane. I think asm, C, python is good, but wish we could do something with a functional language, particularly haskell, to show the power of a compiler. but this could be an entire module.
