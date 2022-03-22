DATA := data/
OUTPUT := output/
PREDICCION := predicciones/

# Estilos que se utilizan para mostrar mensajes en el terminal
GREEN = \033[0;32m
CYAN = \033[0;36m
NO_COLOR = \033[0m
BOLD = \033[1m

all: mkdir compilar ayuda_ayuda_retro ejecuta_problema_real6

mkdir:
	@mkdir -p $(OUTPUT) $(PREDICCION)

compilar:
	@echo "${GREEN}Compilando todos los archivos relacionados con Perceptron Multicapa.\n${NO_COLOR}"

ayuda_ayuda_retro:
	@echo "${GREEN}Uso del fichero multicapa.py.${NO_COLOR}\n"
	@echo "${GREEN}Existe 3 modos de operacion ${CYAN}--modo1 {fichero} {porcion}; --modo2 {fichero}; --modo3 {train_file} {test_file}.${NO_COLOR}\n"
	@echo "${GREEN}Otros argumentos disponibles:${NO_COLOR}\n"
	@echo "${GREEN}Arg ${CYAN}--f_out {nombre} ${GREEN}para indicar una salida, sino se imprime directamente por la pantalla.${NO_COLOR}\n"
	@echo "${GREEN}Arg ${CYAN}--alpha {float} ${GREEN}para indicar la tasa de aprendizaje, por defecto es 0,1.${NO_COLOR}\n"
	@echo "${GREEN}Arg ${CYAN}--norm {true/false} ${GREEN}para normalizar o no.${NO_COLOR}\n"
	@echo "${GREEN}Arg ${CYAN}--torelancia {float} ${GREEN}para indicar con cuanta tolerancia admite para parar la ejecución.${NO_COLOR}\n"
	@echo "${GREEN}Arg ${CYAN}--epoca {int} ${GREEN}para indicar cuantas épocas se va a ejecutar como máximo.${NO_COLOR}\n"
	@echo "${GREEN}Arg ${CYAN}--neu {int, int, etc}${GREEN}, se indica con una lista de números de neuronas de cada capa oculta separado por como.${NO_COLOR}\n"
	@echo "${GREEN}Un ejemplo del uso sería ${CYAN}python3 multicapa.py --modo1 $(DATA)problema_real1.txt 0.25 --alpha 0.1 --neu 2 --f_out $(OUTPUT)salida${NO_COLOR}\n"

ejecuta_problema_real6:
	@echo "Ejecutando el script Multicapa."
	python3 multicapaSinParar.py --modo1 $(DATA)problema_real6.txt 0.3 --f_out $(PREDICCION)problema_real6_mul.txt --alpha 0.1 --epoca 5000 --neu 20 --norm true
	@echo ""

clean:
	rm -df salida.txt $(OUTPUT)