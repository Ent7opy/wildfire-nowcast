FROM node:20-slim

WORKDIR /app/ui

COPY package*.json ./
RUN npm ci

COPY . .

EXPOSE 8501

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "8501"]
